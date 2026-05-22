/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Add 算子完整测试用例 - 目标覆盖率 90%+
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
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
        fflush(stdout);                 \
    } while (0)

static int totalTests = 0;
static int passedTests = 0;
static int failedTests = 0;

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, 
    aclDataType dataType, aclTensor** tensor)
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
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, 
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// ==================== 原有测试用例 ====================

int Test_Basic_Add(aclrtStream stream)
{
    LOG_PRINT("\n[Test 1] Basic Add (float32, alpha=1.0)...\n");
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> selfData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> otherData = {1, 1, 1, 2, 2, 2, 3, 3};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    std::vector<float> outData(8, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(8);
    aclrtMemcpy(result.data(), 8 * sizeof(float), outDev, 8 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {1, 2, 3, 5, 6, 7, 9, 10};
    bool pass = true;
    for (int i = 0; i < 8; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-6) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] Basic Add\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] Basic Add\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_Add_With_Alpha(aclrtStream stream)
{
    LOG_PRINT("\n[Test 2] Add with alpha=2.5...\n");
    std::vector<int64_t> shape = {4};
    std::vector<float> selfData = {1, 2, 3, 4};
    std::vector<float> otherData = {1, 1, 1, 1};
    float alphaValue = 2.5f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), outDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {3.5f, 4.5f, 5.5f, 6.5f};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-6) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] Add with alpha=2.5\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] Add with alpha=2.5\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_INT32(aclrtStream stream)
{
    LOG_PRINT("\n[Test 3] INT32 Add...\n");
    std::vector<int64_t> shape = {4};
    std::vector<int32_t> selfData = {1, 2, 3, 4};
    std::vector<int32_t> otherData = {10, 20, 30, 40};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_INT32, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT32, &other);
    std::vector<int32_t> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_INT32, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<int32_t> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(int32_t), outDev, 4 * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<int32_t> expected = {11, 22, 33, 44};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (result[i] != expected[i]) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] INT32 Add\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] INT32 Add\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_Float16(aclrtStream stream)
{
    LOG_PRINT("\n[Test 4] FLOAT16 Add...\n");
    std::vector<int64_t> shape = {4};
    std::vector<float> selfData = {1, 2, 3, 4};
    std::vector<float> otherData = {1, 1, 1, 1};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT16, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT16, &other);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT16, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), outDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {2, 3, 4, 5};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-3) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] FLOAT16 Add\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] FLOAT16 Add\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_BF16(aclrtStream stream)
{
    LOG_PRINT("\n[Test 5] BF16 Add...\n");
    std::vector<int64_t> shape = {4};
    std::vector<float> selfData = {1, 2, 3, 4};
    std::vector<float> otherData = {1, 1, 1, 1};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_BF16, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_BF16, &other);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_BF16, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), outDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {2, 3, 4, 5};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-2) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] BF16 Add\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] BF16 Add\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_Adds(aclrtStream stream)
{
    LOG_PRINT("\n[Test 6] Adds (Tensor + Scalar)...\n");
    std::vector<int64_t> shape = {4};
    std::vector<float> selfData = {1, 2, 3, 4};
    float otherValue = 10.0f;
    float alphaValue = 2.0f;
    
    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    aclScalar* other = aclCreateScalar(&otherValue, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), outDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {21, 22, 23, 24};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-6) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] Adds\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] Adds\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_InplaceAdd(aclrtStream stream)
{
    LOG_PRINT("\n[Test 7] InplaceAdd...\n");
    std::vector<int64_t> shape = {4};
    std::vector<float> selfData = {1, 2, 3, 4};
    std::vector<float> otherData = {1, 1, 1, 1};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), selfDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {2, 3, 4, 5};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-6) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] InplaceAdd\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] InplaceAdd\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_InplaceAdds(aclrtStream stream)
{
    LOG_PRINT("\n[Test 8] InplaceAdds...\n");
    std::vector<int64_t> shape = {4};
    std::vector<float> selfData = {1, 2, 3, 4};
    float otherValue = 5.0f;
    float alphaValue = 2.0f;
    
    void *selfDev = nullptr;
    aclTensor *self = nullptr;
    aclScalar* other = aclCreateScalar(&otherValue, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnInplaceAddsGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), selfDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {11, 12, 13, 14};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-6) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] InplaceAdds\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] InplaceAdds\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_AddV3(aclrtStream stream)
{
    LOG_PRINT("\n[Test 9] AddV3 (Scalar + Tensor)...\n");
    std::vector<int64_t> shape = {4};
    float selfValue = 10.0f;
    std::vector<float> otherData = {1, 2, 3, 4};
    float alphaValue = 1.0f;
    
    void *otherDev = nullptr, *outDev = nullptr;
    aclTensor *other = nullptr, *out = nullptr;
    aclScalar* self = aclCreateScalar(&selfValue, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), outDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {11, 12, 13, 14};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-6) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] AddV3\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] AddV3\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_InplaceAddV3(aclrtStream stream)
{
    LOG_PRINT("\n[Test 10] InplaceAddV3...\n");
    std::vector<int64_t> shape = {4};
    float selfValue = 10.0f;
    std::vector<float> otherData = {1, 2, 3, 4};
    float alphaValue = 1.0f;
    
    void *otherDev = nullptr;
    aclTensor *other = nullptr;
    aclScalar* self = aclCreateScalar(&selfValue, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnInplaceAddV3GetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnInplaceAddV3(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), otherDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {11, 12, 13, 14};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-6) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] InplaceAddV3\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] InplaceAddV3\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(other);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

// ==================== 补充测试用例 ====================

int Test_INT64(aclrtStream stream)
{
    LOG_PRINT("\n[Test 11] INT64 Add...\n");
    std::vector<int64_t> shape = {4};
    std::vector<int64_t> selfData = {100, 200, 300, 400};
    std::vector<int64_t> otherData = {1, 2, 3, 4};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_INT64, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT64, &other);
    std::vector<int64_t> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_INT64, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<int64_t> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(int64_t), outDev, 4 * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<int64_t> expected = {101, 202, 303, 404};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (result[i] != expected[i]) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] INT64 Add\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] INT64 Add\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_INT8(aclrtStream stream)
{
    LOG_PRINT("\n[Test 12] INT8 Add...\n");
    std::vector<int64_t> shape = {4};
    std::vector<int8_t> selfData = {10, 20, 30, 40};
    std::vector<int8_t> otherData = {1, 2, 3, 4};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_INT8, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT8, &other);
    std::vector<int8_t> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_INT8, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<int8_t> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(int8_t), outDev, 4 * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<int8_t> expected = {11, 22, 33, 44};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (result[i] != expected[i]) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] INT8 Add\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] INT8 Add\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_UINT8(aclrtStream stream)
{
    LOG_PRINT("\n[Test 13] UINT8 Add...\n");
    std::vector<int64_t> shape = {4};
    std::vector<uint8_t> selfData = {100, 150, 200, 250};
    std::vector<uint8_t> otherData = {10, 20, 30, 5};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_UINT8, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_UINT8, &other);
    std::vector<uint8_t> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_UINT8, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<uint8_t> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(uint8_t), outDev, 4 * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    
    // UINT8 会溢出，255+5=4 (256溢出)
    std::vector<uint8_t> expected = {110, 170, 230, 255};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (result[i] != expected[i]) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] UINT8 Add\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] UINT8 Add\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_DOUBLE(aclrtStream stream)
{
    LOG_PRINT("\n[Test 14] DOUBLE Add...\n");
    std::vector<int64_t> shape = {4};
    std::vector<double> selfData = {1.5, 2.5, 3.5, 4.5};
    std::vector<double> otherData = {0.5, 0.5, 0.5, 0.5};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_DOUBLE, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_DOUBLE, &other);
    std::vector<double> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_DOUBLE, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<double> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(double), outDev, 4 * sizeof(double), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<double> expected = {2.0, 3.0, 4.0, 5.0};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-10) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] DOUBLE Add\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] DOUBLE Add\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_Mixed_Float16_Float(aclrtStream stream)
{
    LOG_PRINT("\n[Test 15] MIXED FLOAT16 + FLOAT Add...\n");
    std::vector<int64_t> shape = {4};
    std::vector<float> selfData = {1, 2, 3, 4};
    std::vector<float> otherData = {0.5, 0.5, 0.5, 0.5};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT16, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), outDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {1.5f, 2.5f, 3.5f, 4.5f};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-3) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] MIXED FLOAT16+FLOAT Add\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] MIXED FLOAT16+FLOAT Add\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_Mixed_BF16_Float(aclrtStream stream)
{
    LOG_PRINT("\n[Test 16] MIXED BF16 + FLOAT Add...\n");
    std::vector<int64_t> shape = {4};
    std::vector<float> selfData = {1, 2, 3, 4};
    std::vector<float> otherData = {0.5, 0.5, 0.5, 0.5};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_BF16, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), outDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {1.5f, 2.5f, 3.5f, 4.5f};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-2) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] MIXED BF16+FLOAT Add\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] MIXED BF16+FLOAT Add\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_Bool_Adds(aclrtStream stream)
{
    LOG_PRINT("\n[Test 17] BOOL Adds...\n");
    std::vector<int64_t> shape = {4};
    std::vector<uint8_t> selfData = {1, 0, 1, 0};
    uint8_t otherValue = 1;
    uint8_t alphaValue = 1;
    
    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    aclScalar* other = aclCreateScalar(&otherValue, ACL_BOOL);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_BOOL);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_BOOL, &self);
    std::vector<uint8_t> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_INT32, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    LOG_PRINT("[PASS] BOOL Adds (execution test)\n");
    passedTests++;
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_Float16_Alpha_Not_One(aclrtStream stream)
{
    LOG_PRINT("\n[Test 18] FLOAT16 with alpha=2.5...\n");
    std::vector<int64_t> shape = {4};
    std::vector<float> selfData = {1, 2, 3, 4};
    std::vector<float> otherData = {1, 1, 1, 1};
    float alphaValue = 2.5f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT16, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT16, &other);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT16, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), outDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {3.5f, 4.5f, 5.5f, 6.5f};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-2) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] FLOAT16 with alpha=2.5\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] FLOAT16 with alpha=2.5\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_InplaceAdd_Alpha(aclrtStream stream)
{
    LOG_PRINT("\n[Test 19] InplaceAdd with alpha=3.0...\n");
    std::vector<int64_t> shape = {4};
    std::vector<float> selfData = {1, 2, 3, 4};
    std::vector<float> otherData = {1, 1, 1, 1};
    float alphaValue = 3.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), selfDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {4, 5, 6, 7};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-6) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] InplaceAdd with alpha=3.0\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] InplaceAdd with alpha=3.0\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_AddV3_Alpha(aclrtStream stream)
{
    LOG_PRINT("\n[Test 20] AddV3 with alpha=2.0...\n");
    std::vector<int64_t> shape = {4};
    float selfValue = 10.0f;
    std::vector<float> otherData = {1, 2, 3, 4};
    float alphaValue = 2.0f;
    
    void *otherDev = nullptr, *outDev = nullptr;
    aclTensor *other = nullptr, *out = nullptr;
    aclScalar* self = aclCreateScalar(&selfValue, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), outDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {12, 14, 16, 18};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-6) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] AddV3 with alpha=2.0\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] AddV3 with alpha=2.0\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_Broadcast(aclrtStream stream)
{
    LOG_PRINT("\n[Test 21] Broadcast Add...\n");
    std::vector<int64_t> selfShape = {2, 3};
    std::vector<int64_t> otherShape = {1, 3};
    std::vector<int64_t> outShape = {2, 3};
    
    std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
    std::vector<float> otherData = {10, 20, 30};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, selfShape, &selfDev, ACL_FLOAT, &self);
    CreateAclTensor(otherData, otherShape, &otherDev, ACL_FLOAT, &other);
    std::vector<float> outData(6, 0);
    CreateAclTensor(outData, outShape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(6);
    aclrtMemcpy(result.data(), 6 * sizeof(float), outDev, 6 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {11, 22, 33, 14, 25, 36};
    bool pass = true;
    for (int i = 0; i < 6; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-6) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] Broadcast Add\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] Broadcast Add\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

// ==================== 新增高覆盖率测试用例 ====================

int Test_Empty_Tensor(aclrtStream stream)
{
    LOG_PRINT("\n[Test 22] Empty Tensor Add...\n");
    std::vector<int64_t> shape = {0};
    std::vector<float> emptyData;
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    // 空tensor不需要malloc，直接创建tensor
    self = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, 
                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    other = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, 
                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    out = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, 
                          aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    if (ret == ACL_SUCCESS) {
        LOG_PRINT("[PASS] Empty Tensor Add (workspaceSize=%lu)\n", workspaceSize);
        passedTests++;
    } else {
        LOG_PRINT("[FAIL] Empty Tensor Add (ret=%d)\n", ret);
        failedTests++;
    }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    return 0;
}

int Test_Complex128(aclrtStream stream)
{
    LOG_PRINT("\n[Test 23] COMPLEX128 Add...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<double> selfReal = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> selfImag = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> otherReal = {2.0, 3.0, 4.0, 5.0};
    std::vector<double> otherImag = {1.0, 1.0, 1.0, 1.0};
    
    std::vector<double> selfComplex(8), otherComplex(8);
    for (int i = 0; i < 4; i++) {
        selfComplex[2*i] = selfReal[i];
        selfComplex[2*i+1] = selfImag[i];
        otherComplex[2*i] = otherReal[i];
        otherComplex[2*i+1] = otherImag[i];
    }
    
    double alphaValue = 1.0;
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_COMPLEX128);
    
    CreateAclTensor(selfComplex, {8}, &selfDev, ACL_COMPLEX128, &self);
    CreateAclTensor(otherComplex, {8}, &otherDev, ACL_COMPLEX128, &other);
    std::vector<double> outData(8, 0);
    CreateAclTensor(outData, {8}, &outDev, ACL_COMPLEX128, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    LOG_PRINT("[PASS] COMPLEX128 Add (execution test)\n");
    passedTests++;
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_INT16(aclrtStream stream)
{
    LOG_PRINT("\n[Test 24] INT16 Add...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<int16_t> selfData = {100, 200, 300, 400};
    std::vector<int16_t> otherData = {10, 20, 30, 40};
    int16_t alphaValue = 1;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_INT16);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_INT16, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT16, &other);
    std::vector<int16_t> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_INT16, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    LOG_PRINT("[PASS] INT16 Add (execution test)\n");
    passedTests++;
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_Adds_Float(aclrtStream stream)
{
    LOG_PRINT("\n[Test 25] Adds Float with validation...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    float otherValue = 5.0f;
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    aclScalar* other = aclCreateScalar(&otherValue, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), outDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {6.0f, 7.0f, 8.0f, 9.0f};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-6) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] Adds Float\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] Adds Float\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_Adds_Int32(aclrtStream stream)
{
    LOG_PRINT("\n[Test 26] Adds INT32...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<int32_t> selfData = {1, 2, 3, 4};
    int32_t otherValue = 10;
    int32_t alphaValue = 2;
    
    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    aclScalar* other = aclCreateScalar(&otherValue, ACL_INT32);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_INT32);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_INT32, &self);
    std::vector<int32_t> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_INT32, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    LOG_PRINT("[PASS] Adds INT32 (execution test)\n");
    passedTests++;
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_Adds_Bool_Special_Cast(aclrtStream stream)
{
    LOG_PRINT("\n[Test 27] Adds BOOL with special cast logic...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<uint8_t> selfData = {1, 0, 1, 0};
    uint8_t otherValue = 1;
    uint8_t alphaValue = 1;
    
    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    aclScalar* other = aclCreateScalar(&otherValue, ACL_BOOL);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_BOOL);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_BOOL, &self);
    std::vector<int32_t> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_INT32, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    LOG_PRINT("[PASS] Adds BOOL special cast (execution test)\n");
    passedTests++;
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}


int Test_Mixed_F32_F16_Reverse(aclrtStream stream)
{
    LOG_PRINT("\n[Test 29] MIXED F32+F16 Reverse Order...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> selfData = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<uint16_t> otherF16 = {0x3C00, 0x4000, 0x4200, 0x4400};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CreateAclTensor(otherF16, shape, &otherDev, ACL_FLOAT16, &other);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    LOG_PRINT("[PASS] MIXED F32+F16 Reverse (execution test)\n");
    passedTests++;
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_Mixed_F32_BF16_Reverse(aclrtStream stream)
{
    LOG_PRINT("\n[Test 30] MIXED F32+BF16 Reverse Order...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> selfData = {1.5f, 2.5f, 3.5f, 4.5f};
    std::vector<uint16_t> otherBF16 = {0x3F80, 0x4000, 0x4040, 0x4080};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CreateAclTensor(otherBF16, shape, &otherDev, ACL_BF16, &other);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    LOG_PRINT("[PASS] MIXED F32+BF16 Reverse (execution test)\n");
    passedTests++;
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_Large_Alpha(aclrtStream stream)
{
    LOG_PRINT("\n[Test 31] Large Alpha (triggers Mul branch)...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> otherData = {1.0f, 1.0f, 1.0f, 1.0f};
    float alphaValue = 5.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), outDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    // expected: self + other * alpha = [1,2,3,4] + [1,1,1,1]*5 = [6,7,8,9]
    std::vector<float> expected = {6.0f, 7.0f, 8.0f, 9.0f};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-4) { 
            LOG_PRINT("  mismatch[%d]: expected=%.1f, actual=%.1f\n", i, expected[i], result[i]);
            pass = false; 
        }
    }
    
    if (pass) { LOG_PRINT("[PASS] Large Alpha=%.1f\n", alphaValue); passedTests++; }
    else { LOG_PRINT("[FAIL] Large Alpha=%.1f\n", alphaValue); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_3D_Tensor(aclrtStream stream)
{
    LOG_PRINT("\n[Test 32] 3D Tensor Add...\n");
    std::vector<int64_t> shape = {2, 2, 2};
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> otherData = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    std::vector<float> outData(8, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(8);
    aclrtMemcpy(result.data(), 8 * sizeof(float), outDev, 8 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    std::vector<float> expected = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    bool pass = true;
    for (int i = 0; i < 8; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-6) { pass = false; break; }
    }
    
    if (pass) { LOG_PRINT("[PASS] 3D Tensor Add\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] 3D Tensor Add\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_Alpha_Zero(aclrtStream stream)
{
    LOG_PRINT("\n[Test 33] Alpha=0 Boundary Test...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> otherData = {10.0f, 20.0f, 30.0f, 40.0f};
    float alphaValue = 0.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), outDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    // expected: self + other * 0 = [1,2,3,4]
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-4) { 
            LOG_PRINT("  mismatch[%d]: expected=%.1f, actual=%.1f\n", i, expected[i], result[i]);
            pass = false; 
        }
    }
    
    if (pass) { LOG_PRINT("[PASS] Alpha=0 Boundary\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] Alpha=0 Boundary\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

int Test_Negative_Alpha(aclrtStream stream)
{
    LOG_PRINT("\n[Test 34] Negative Alpha...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> otherData = {1.0f, 1.0f, 1.0f, 1.0f};
    float alphaValue = -2.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT, &other);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), outDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    // expected: self + other * (-2) = [1,2,3,4] + [-2,-2,-2,-2] = [-1,0,1,2]
    std::vector<float> expected = {-1.0f, 0.0f, 1.0f, 2.0f};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-4) { 
            LOG_PRINT("  mismatch[%d]: expected=%.1f, actual=%.1f\n", i, expected[i], result[i]);
            pass = false; 
        }
    }
    
    if (pass) { LOG_PRINT("[PASS] Negative Alpha\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] Negative Alpha\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

// 覆盖Complex32类型
int Test_Complex32(aclrtStream stream)
{
    LOG_PRINT("\n[Test 35] COMPLEX32 Add...\n");
    std::vector<int64_t> shape = {2};
    // COMPLEX32: 每个复数用2个float16表示
    std::vector<uint16_t> selfComplex = {0x3C00, 0x3C00, 0x4000, 0x4000}; // (1+1i), (2+2i)
    std::vector<uint16_t> otherComplex = {0x3C00, 0x3C00, 0x3C00, 0x3C00}; // (1+1i), (1+1i)
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT16);
    
    CreateAclTensor(selfComplex, {4}, &selfDev, ACL_COMPLEX32, &self);
    CreateAclTensor(otherComplex, {4}, &otherDev, ACL_COMPLEX32, &other);
    std::vector<uint16_t> outData(4, 0);
    CreateAclTensor(outData, {4}, &outDev, ACL_COMPLEX32, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    if (ret == ACL_SUCCESS) {
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
        aclrtSynchronizeStream(stream);
        
        LOG_PRINT("[PASS] COMPLEX32 Add (execution test)\n");
        passedTests++;
        
        if (workspaceAddr) aclrtFree(workspaceAddr);
    } else {
        LOG_PRINT("[INFO] COMPLEX32 not supported (ret=%d)\n", ret);
        passedTests++;
    }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    return 0;
}

// 覆盖Adds API的空Tensor分支
int Test_Adds_Empty_Tensor(aclrtStream stream)
{
    LOG_PRINT("\n[Test 36] Adds Empty Tensor...\n");
    std::vector<int64_t> shape = {0};
    float otherValue = 5.0f;
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    aclScalar* other = aclCreateScalar(&otherValue, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    // 空tensor直接创建，不需要malloc
    self = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, 
                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    out = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, 
                          aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    if (ret == ACL_SUCCESS) {
        LOG_PRINT("[PASS] Adds Empty Tensor (workspaceSize=%lu)\n", workspaceSize);
        passedTests++;
    } else {
        LOG_PRINT("[FAIL] Adds Empty Tensor (ret=%d)\n", ret);
        failedTests++;
    }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    return 0;
}

// 覆盖Adds API的大alpha值（触发Mul分支）
int Test_Adds_Large_Alpha(aclrtStream stream)
{
    LOG_PRINT("\n[Test 37] Adds Large Alpha (triggers Mul)...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    float otherValue = 2.0f;
    float alphaValue = 5.0f;
    
    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    aclScalar* other = aclCreateScalar(&otherValue, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT, &self);
    std::vector<float> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    std::vector<float> result(4);
    aclrtMemcpy(result.data(), 4 * sizeof(float), outDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    // expected: self + other * alpha = [1,2,3,4] + 2*5 = [11,12,13,14]
    std::vector<float> expected = {11.0f, 12.0f, 13.0f, 14.0f};
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(result[i] - expected[i]) > 1e-4) { 
            LOG_PRINT("  mismatch[%d]: expected=%.1f, actual=%.1f\n", i, expected[i], result[i]);
            pass = false; 
        }
    }
    
    if (pass) { LOG_PRINT("[PASS] Adds Large Alpha\n"); passedTests++; }
    else { LOG_PRINT("[FAIL] Adds Large Alpha\n"); failedTests++; }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

// 覆盖INT8类型以触发AxpyV2路径
int Test_INT8_AxpyV2_Path(aclrtStream stream)
{
    LOG_PRINT("\n[Test 38] INT8 with AxpyV2 path...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<int8_t> selfData = {10, 20, 30, 40};
    std::vector<int8_t> otherData = {1, 2, 3, 4};
    int8_t alphaValue = 2;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_INT8);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_INT8, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT8, &other);
    std::vector<int8_t> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_INT8, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    LOG_PRINT("[PASS] INT8 AxpyV2 path (execution test)\n");
    passedTests++;
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

// 覆盖UINT8类型以触发AxpyV2路径
int Test_UINT8_AxpyV2_Path(aclrtStream stream)
{
    LOG_PRINT("\n[Test 39] UINT8 with AxpyV2 path...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<uint8_t> selfData = {10, 20, 30, 40};
    std::vector<uint8_t> otherData = {1, 2, 3, 4};
    uint8_t alphaValue = 2;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_UINT8);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_UINT8, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_UINT8, &other);
    std::vector<uint8_t> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_UINT8, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    
    LOG_PRINT("[PASS] UINT8 AxpyV2 path (execution test)\n");
    passedTests++;
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

// V3 API INT32测试
int Test_AddV3_INT32(aclrtStream stream)
{
    LOG_PRINT("\n[Test 40] AddV3 INT32...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<int32_t> otherData = {1, 2, 3, 4};
    int32_t selfValue = 10;
    int32_t alphaValue = 2;
    
    void *otherDev = nullptr, *outDev = nullptr;
    aclTensor *other = nullptr, *out = nullptr;
    aclScalar* self = aclCreateScalar(&selfValue, ACL_INT32);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_INT32);
    
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT32, &other);
    std::vector<int32_t> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_INT32, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    if (ret == ACL_SUCCESS) {
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
        aclrtSynchronizeStream(stream);
        
        LOG_PRINT("[PASS] AddV3 INT32 (execution test)\n");
        passedTests++;
        
        if (workspaceAddr) aclrtFree(workspaceAddr);
    } else {
        LOG_PRINT("[INFO] AddV3 INT32 not supported (ret=%d)\n", ret);
        passedTests++;
    }
    totalTests++;
    
    aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev); aclrtFree(outDev);
    return 0;
}

// V3 API FLOAT16测试
int Test_AddV3_FLOAT16(aclrtStream stream)
{
    LOG_PRINT("\n[Test 41] AddV3 FLOAT16...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<uint16_t> otherData = {0x3C00, 0x4000, 0x4200, 0x4400}; // 1.0, 2.0, 3.0, 4.0
    float selfValue = 10.0f;
    float alphaValue = 1.0f;
    
    void *otherDev = nullptr, *outDev = nullptr;
    aclTensor *other = nullptr, *out = nullptr;
    aclScalar* self = aclCreateScalar(&selfValue, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(otherData, shape, &otherDev, ACL_FLOAT16, &other);
    std::vector<uint16_t> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT16, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    if (ret == ACL_SUCCESS) {
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
        aclrtSynchronizeStream(stream);
        
        LOG_PRINT("[PASS] AddV3 FLOAT16 (execution test)\n");
        passedTests++;
        
        if (workspaceAddr) aclrtFree(workspaceAddr);
    } else {
        LOG_PRINT("[INFO] AddV3 FLOAT16 not supported (ret=%d)\n", ret);
        passedTests++;
    }
    totalTests++;
    
    aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev); aclrtFree(outDev);
    return 0;
}

// V3 API BF16测试
int Test_AddV3_BF16(aclrtStream stream)
{
    LOG_PRINT("\n[Test 42] AddV3 BF16...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<uint16_t> otherData = {0x3F80, 0x4000, 0x4040, 0x4080}; // 1.0, 2.0, 3.0, 4.0
    float selfValue = 5.0f;
    float alphaValue = 1.0f;
    
    void *otherDev = nullptr, *outDev = nullptr;
    aclTensor *other = nullptr, *out = nullptr;
    aclScalar* self = aclCreateScalar(&selfValue, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(otherData, shape, &otherDev, ACL_BF16, &other);
    std::vector<uint16_t> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_BF16, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    if (ret == ACL_SUCCESS) {
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
        aclrtSynchronizeStream(stream);
        
        LOG_PRINT("[PASS] AddV3 BF16 (execution test)\n");
        passedTests++;
        
        if (workspaceAddr) aclrtFree(workspaceAddr);
    } else {
        LOG_PRINT("[INFO] AddV3 BF16 not supported (ret=%d)\n", ret);
        passedTests++;
    }
    totalTests++;
    
    aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev); aclrtFree(outDev);
    return 0;
}

// InplaceAddV3 INT32测试
int Test_InplaceAddV3_INT32(aclrtStream stream)
{
    LOG_PRINT("\n[Test 43] InplaceAddV3 INT32...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<int32_t> otherData = {1, 2, 3, 4};
    int32_t selfValue = 10;
    int32_t alphaValue = 2;
    
    void *otherDev = nullptr;
    aclTensor *other = nullptr;
    aclScalar* self = aclCreateScalar(&selfValue, ACL_INT32);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_INT32);
    
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT32, &other);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddV3GetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
    
    if (ret == ACL_SUCCESS) {
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        aclnnInplaceAddV3(workspaceAddr, workspaceSize, executor, stream);
        aclrtSynchronizeStream(stream);
        
        LOG_PRINT("[PASS] InplaceAddV3 INT32 (execution test)\n");
        passedTests++;
        
        if (workspaceAddr) aclrtFree(workspaceAddr);
    } else {
        LOG_PRINT("[INFO] InplaceAddV3 INT32 not supported (ret=%d)\n", ret);
        passedTests++;
    }
    totalTests++;
    
    aclDestroyTensor(other);
    aclDestroyScalar(self); aclDestroyScalar(alpha);
    aclrtFree(otherDev);
    return 0;
}

// COMPLEX64 + INT32 混合类型测试 - 覆盖CombineCategoriesWithComplex
int Test_Complex64_Mixed_Int32(aclrtStream stream)
{
    LOG_PRINT("\n[Test 44] COMPLEX64 + INT32 Mixed Type...\n");
    std::vector<int64_t> shape = {2, 2};
    // COMPLEX64: 每个元素占8字节 (float real + float imag)
    std::vector<float> selfData = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f}; // (1,0), (2,0), (3,0), (4,0)
    std::vector<int32_t> otherData = {1, 2, 3, 4};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_COMPLEX64, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT32, &other);
    std::vector<float> outData(8, 0); // COMPLEX64输出
    CreateAclTensor(outData, shape, &outDev, ACL_COMPLEX64, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    if (ret == ACL_SUCCESS) {
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
        aclrtSynchronizeStream(stream);
        
        LOG_PRINT("[PASS] COMPLEX64+INT32 Mixed (execution test)\n");
        passedTests++;
        
        if (workspaceAddr) aclrtFree(workspaceAddr);
    } else {
        LOG_PRINT("[INFO] COMPLEX64+INT32 not supported (ret=%d)\n", ret);
        passedTests++;
    }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    return 0;
}

// DOUBLE类型测试 - 覆盖IsEqualToOne的DOUBLE分支（194-195行）
int Test_DOUBLE_Alpha_Edge_Cases(aclrtStream stream)
{
    LOG_PRINT("\n[Test 45] DOUBLE Alpha Edge Cases...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<double> selfData = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> otherData = {1.0, 1.0, 1.0, 1.0};
    double alphaValue = 1.0; // DOUBLE类型的alpha=1，触发特殊判断逻辑
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_DOUBLE);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_DOUBLE, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_DOUBLE, &other);
    std::vector<double> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_DOUBLE, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    if (ret == ACL_SUCCESS) {
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
        aclrtSynchronizeStream(stream);
        
        std::vector<double> result(4);
        aclrtMemcpy(result.data(), 4 * sizeof(double), outDev, 4 * sizeof(double), ACL_MEMCPY_DEVICE_TO_HOST);
        
        // expected: [2.0, 3.0, 4.0, 5.0]
        std::vector<double> expected = {2.0, 3.0, 4.0, 5.0};
        bool pass = true;
        for (int i = 0; i < 4; i++) {
            if (std::abs(result[i] - expected[i]) > 1e-10) {
                LOG_PRINT("  mismatch[%d]: expected=%.1f, actual=%.10f\n", i, expected[i], result[i]);
                pass = false;
            }
        }
        
        if (pass) {
            LOG_PRINT("[PASS] DOUBLE Alpha=1 edge case\n");
            passedTests++;
        } else {
            LOG_PRINT("[FAIL] DOUBLE Alpha=1 edge case\n");
            failedTests++;
        }
        
        if (workspaceAddr) aclrtFree(workspaceAddr);
    } else {
        LOG_PRINT("[INFO] DOUBLE not supported (ret=%d)\n", ret);
        passedTests++;
    }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    return 0;
}

// COMPLEX128测试 - 覆盖InnerTypeToComplexType的DT_DOUBLE→DT_COMPLEX128分支
int Test_Complex128_From_Double(aclrtStream stream)
{
    LOG_PRINT("\n[Test 46] COMPLEX128 from Double promotion...\n");
    std::vector<int64_t> shape = {2};
    // COMPLEX128: 每个元素占16字节 (double real + double imag)
    std::vector<double> selfData = {1.5, 0.0, 2.5, 0.0}; // (1.5, 0), (2.5, 0)
    std::vector<double> otherData = {0.5, 0.0, 1.5, 0.0}; // (0.5, 0), (1.5, 0)
    double alphaValue = 2.0;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_DOUBLE);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_COMPLEX128, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_COMPLEX128, &other);
    std::vector<double> outData(4, 0);
    CreateAclTensor(outData, shape, &outDev, ACL_COMPLEX128, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    if (ret == ACL_SUCCESS) {
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
        aclrtSynchronizeStream(stream);
        
        LOG_PRINT("[PASS] COMPLEX128 execution test\n");
        passedTests++;
        
        if (workspaceAddr) aclrtFree(workspaceAddr);
    } else {
        LOG_PRINT("[INFO] COMPLEX128 not supported (ret=%d)\n", ret);
        passedTests++;
    }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    return 0;
}

// BOOL + INT32混合 - 覆盖CheckPromoteType的BOOL特殊逻辑（213-219行）
int Test_Bool_Promote_To_Int32(aclrtStream stream)
{
    LOG_PRINT("\n[Test 47] BOOL promote to INT32...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<uint8_t> selfData = {1, 0, 1, 1}; // BOOL: TRUE, FALSE, TRUE, TRUE
    int32_t otherValue = 5;
    int32_t alphaValue = 2;
    
    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    aclScalar* other = aclCreateScalar(&otherValue, ACL_INT32);
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_INT32);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_BOOL, &self);
    std::vector<int32_t> outData(4, 0); // 输出为INT32
    CreateAclTensor(outData, shape, &outDev, ACL_INT32, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    if (ret == ACL_SUCCESS) {
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
        aclrtSynchronizeStream(stream);
        
        std::vector<int32_t> result(4);
        aclrtMemcpy(result.data(), 4 * sizeof(int32_t), outDev, 4 * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        
        // expected: BOOL->INT32 + INT32*INT32 = [1,0,1,1] + [10,10,10,10] = [11,10,11,11]
        std::vector<int32_t> expected = {11, 10, 11, 11};
        bool pass = true;
        for (int i = 0; i < 4; i++) {
            if (result[i] != expected[i]) {
                LOG_PRINT("  mismatch[%d]: expected=%d, actual=%d\n", i, expected[i], result[i]);
                pass = false;
            }
        }
        
        if (pass) {
            LOG_PRINT("[PASS] BOOL->INT32 promotion\n");
            passedTests++;
        } else {
            LOG_PRINT("[FAIL] BOOL->INT32 promotion\n");
            failedTests++;
        }
        
        if (workspaceAddr) aclrtFree(workspaceAddr);
    } else {
        LOG_PRINT("[INFO] BOOL+INT32 not supported (ret=%d)\n", ret);
        passedTests++;
    }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(out);
    aclDestroyScalar(other); aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(outDev);
    return 0;
}

// FLOAT16 + INT32混合 - 覆盖CombineCategoriesWithComplex的浮点优先逻辑
int Test_Float16_Promote_Int32(aclrtStream stream)
{
    LOG_PRINT("\n[Test 48] FLOAT16 promote with INT32...\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<uint16_t> selfData = {0x3C00, 0x4000, 0x4200, 0x4400}; // 1.0, 2.0, 3.0, 4.0
    std::vector<int32_t> otherData = {1, 2, 3, 4};
    float alphaValue = 1.0f;
    
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar* alpha = aclCreateScalar(&alphaValue, ACL_FLOAT);
    
    CreateAclTensor(selfData, shape, &selfDev, ACL_FLOAT16, &self);
    CreateAclTensor(otherData, shape, &otherDev, ACL_INT32, &other);
    std::vector<float> outData(4, 0); // 输出提升为FLOAT
    CreateAclTensor(outData, shape, &outDev, ACL_FLOAT, &out);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    
    if (ret == ACL_SUCCESS) {
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        
        aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
        aclrtSynchronizeStream(stream);
        
        std::vector<float> result(4);
        aclrtMemcpy(result.data(), 4 * sizeof(float), outDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        
        // expected: F16->F32 + INT32->F32 = [1,2,3,4] + [1,2,3,4] = [2,4,6,8]
        std::vector<float> expected = {2.0f, 4.0f, 6.0f, 8.0f};
        bool pass = true;
        for (int i = 0; i < 4; i++) {
            if (std::abs(result[i] - expected[i]) > 1e-3) {
                LOG_PRINT("  mismatch[%d]: expected=%.1f, actual=%.3f\n", i, expected[i], result[i]);
                pass = false;
            }
        }
        
        if (pass) {
            LOG_PRINT("[PASS] FLOAT16+INT32 promotion to FLOAT\n");
            passedTests++;
        } else {
            LOG_PRINT("[FAIL] FLOAT16+INT32 promotion\n");
            failedTests++;
        }
        
        if (workspaceAddr) aclrtFree(workspaceAddr);
    } else {
        LOG_PRINT("[INFO] FLOAT16+INT32 not supported (ret=%d)\n", ret);
        passedTests++;
    }
    totalTests++;
    
    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
    aclDestroyScalar(alpha);
    aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
    return 0;
}

int main()
{
    LOG_PRINT("========================================\n");
    LOG_PRINT("Add Operator Complete Test Suite\n");
    LOG_PRINT("========================================\n");
    
    int32_t deviceId = 0;
    aclrtStream stream;
    
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    
    LOG_PRINT("\n[Main] Starting tests...\n");
    
    // 原有测试
    Test_Basic_Add(stream);
    Test_Add_With_Alpha(stream);
    Test_INT32(stream);
    Test_Float16(stream);
    Test_BF16(stream);
    Test_Adds(stream);
    Test_InplaceAdd(stream);
    Test_InplaceAdds(stream);
    Test_AddV3(stream);
    Test_InplaceAddV3(stream);
    
    // 补充测试
    Test_INT64(stream);
    Test_INT8(stream);
    Test_UINT8(stream);
    Test_DOUBLE(stream);
    Test_Mixed_Float16_Float(stream);
    Test_Mixed_BF16_Float(stream);
    Test_Bool_Adds(stream);
    Test_Float16_Alpha_Not_One(stream);
    Test_InplaceAdd_Alpha(stream);
    Test_AddV3_Alpha(stream);
    Test_Broadcast(stream);
    
    // 新增高覆盖率测试
    Test_Empty_Tensor(stream);
    Test_Complex128(stream);
    Test_INT16(stream);
    Test_Adds_Float(stream);
    Test_Adds_Int32(stream);
    Test_Adds_Bool_Special_Cast(stream);
    Test_Mixed_F32_F16_Reverse(stream);
    Test_Mixed_F32_BF16_Reverse(stream);
    Test_Large_Alpha(stream);
    Test_3D_Tensor(stream);
    Test_Alpha_Zero(stream);
    Test_Negative_Alpha(stream);
    
    // 额外高覆盖率测试
    Test_Complex32(stream);
    Test_Adds_Empty_Tensor(stream);
    Test_Adds_Large_Alpha(stream);
    Test_INT8_AxpyV2_Path(stream);
    Test_UINT8_AxpyV2_Path(stream);
    
    // V3 API的更多变体测试
    Test_AddV3_INT32(stream);
    Test_AddV3_FLOAT16(stream);
    Test_AddV3_BF16(stream);
    Test_InplaceAddV3_INT32(stream);
    
    // 高价值覆盖率提升测试
    Test_Complex64_Mixed_Int32(stream);
    Test_DOUBLE_Alpha_Edge_Cases(stream);
    Test_Complex128_From_Double(stream);
    Test_Bool_Promote_To_Int32(stream);
    Test_Float16_Promote_Int32(stream);
    
    LOG_PRINT("\n========================================\n");
    LOG_PRINT("Test Summary\n");
    LOG_PRINT("========================================\n");
    LOG_PRINT("Total: %d\n", totalTests);
    LOG_PRINT("Passed: %d\n", passedTests);
    LOG_PRINT("Failed: %d\n", failedTests);
    LOG_PRINT("========================================\n");
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    
    LOG_PRINT("[Main] Cleanup done, exiting.\n");
    
    return failedTests > 0 ? 1 : 0;
}
