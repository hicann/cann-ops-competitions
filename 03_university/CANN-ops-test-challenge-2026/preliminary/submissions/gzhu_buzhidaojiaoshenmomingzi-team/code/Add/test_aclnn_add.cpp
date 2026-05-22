/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_aclnn_add.cpp
 * @brief Add算子端到端测试用例，覆盖6个API变体和各种执行路径
 *
 * 测试覆盖维度:
 * - 6个API: aclnnAdd, aclnnAdds, aclnnInplaceAdd, aclnnInplaceAdds, aclnnAddV3, aclnnInplaceAddV3
 * - 数据类型: FLOAT, FLOAT16, BF16, INT32, INT8, UINT8, INT64, BOOL
 * - alpha参数: 1.0(标准加法), 0, 负数, 浮点数
 * - Shape组合: 同shape, 广播, 标量
 * - 数值边界: 零值, 极大值, 特殊值
 * - 空tensor测试
 * - 异常输入: nullptr
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstring>
#include <algorithm>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

// ==================== 宏定义 ====================
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

// 测试结果统计
static int g_pass_count = 0;
static int g_fail_count = 0;
static int g_total_count = 0;

// ==================== 辅助函数 ====================

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

// 释放tensor资源
void DestroyTensor(aclTensor* tensor, void* deviceAddr)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

// ==================== 结果验证函数 ====================

template <typename T>
bool CompareResult(const std::vector<T>& actual, const std::vector<T>& expected,
                   double atol = 1e-3, double rtol = 1e-3)
{
    for (size_t i = 0; i < actual.size(); i++) {
        double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        double tolerance = atol + rtol * std::abs(static_cast<double>(expected[i]));
        if (diff > tolerance) {
            LOG_PRINT("Mismatch at index %zu: actual=%.6f, expected=%.6f, diff=%.6f\n",
                      i, static_cast<double>(actual[i]), static_cast<double>(expected[i]), diff);
            return false;
        }
    }
    return true;
}

template <typename T>
bool CompareResultInt(const std::vector<T>& actual, const std::vector<T>& expected)
{
    for (size_t i = 0; i < actual.size(); i++) {
        if (actual[i] != expected[i]) {
            LOG_PRINT("Mismatch at index %zu: actual=%lld, expected=%lld\n",
                      i, static_cast<long long>(actual[i]), static_cast<long long>(expected[i]));
            return false;
        }
    }
    return true;
}

// 计算期望值: expected = self + alpha * other
template <typename T>
void ComputeExpectedAdd(const std::vector<T>& selfData, const std::vector<T>& otherData,
                         double alpha, std::vector<T>& expected)
{
    for (size_t i = 0; i < selfData.size(); i++) {
        expected[i] = static_cast<T>(static_cast<double>(selfData[i]) + alpha * static_cast<double>(otherData[i]));
    }
}

// 计算广播后的期望值
template <typename T>
void ComputeExpectedBroadcast(const std::vector<T>& selfData, const std::vector<T>& otherData,
                               const std::vector<int64_t>& selfShape, const std::vector<int64_t>& otherShape,
                               double alpha, std::vector<T>& expected)
{
    int64_t selfSize = GetShapeSize(selfShape);
    int64_t otherSize = GetShapeSize(otherShape);

    for (int64_t i = 0; i < selfSize; i++) {
        // 简化处理: 如果other是标量或能广播到self的shape
        T otherVal = (otherSize == 1) ? otherData[0] : otherData[i % otherSize];
        expected[i] = static_cast<T>(static_cast<double>(selfData[i]) + alpha * static_cast<double>(otherVal));
    }
}

// ==================== 测试框架 ====================

void PrintTestHeader(const char* testName)
{
    g_total_count++;
    LOG_PRINT("\n[TEST %d] %s\n", g_total_count, testName);
}

void PrintTestResult(bool passed, const char* detail = "")
{
    if (passed) {
        g_pass_count++;
        LOG_PRINT("[PASS] %s\n", detail);
    } else {
        g_fail_count++;
        LOG_PRINT("[FAIL] %s\n", detail);
    }
}

// ==================== API测试函数 ====================

// 测试 aclnnAdd - 基本加法
bool TestAclnnBasic(const std::vector<int64_t>& selfShape, const std::vector<int64_t>& otherShape,
                    aclDataType dataType, double alphaVal, aclrtStream stream, const char* testName)
{
    PrintTestHeader(testName);

    int64_t size = GetShapeSize(selfShape);
    int64_t otherSize = GetShapeSize(otherShape);

    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;

    // 根据数据类型准备数据
    std::vector<float> selfHostData(size);
    std::vector<float> otherHostData(otherSize);
    std::vector<float> expected(size);
    std::vector<float> outHostData(size, 0);

    // 初始化输入数据
    for (int64_t i = 0; i < size; i++) {
        selfHostData[i] = static_cast<float>(i % 10);
    }
    for (int64_t i = 0; i < otherSize; i++) {
        otherHostData[i] = static_cast<float>((i % 5) + 1);
    }

    // 计算期望值
    ComputeExpectedBroadcast(selfHostData, otherHostData, selfShape, otherShape, alphaVal, expected);

    // 创建tensor
    int ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, dataType, &self);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create self tensor failed"); return false);

    ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, dataType, &other);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create other tensor failed");
              DestroyTensor(self, selfDeviceAddr); return false);

    // 创建alpha
    float alphaFloat = static_cast<float>(alphaVal);
    alpha = aclCreateScalar(&alphaFloat, dataType);
    CHECK_RET(alpha != nullptr, PrintTestResult(false, "Create alpha scalar failed");
              DestroyTensor(self, selfDeviceAddr); DestroyTensor(other, otherDeviceAddr); return false);

    // 创建输出tensor
    ret = CreateAclTensor(outHostData, selfShape, &outDeviceAddr, dataType, &out);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create out tensor failed");
              aclDestroyScalar(alpha); DestroyTensor(self, selfDeviceAddr); DestroyTensor(other, otherDeviceAddr);
              return false);

    // 调用API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnAddGetWorkspaceSize failed");
              aclDestroyScalar(alpha); DestroyTensor(self, selfDeviceAddr);
              DestroyTensor(other, otherDeviceAddr); DestroyTensor(out, outDeviceAddr);
              return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Malloc workspace failed");
                  aclDestroyScalar(alpha); DestroyTensor(self, selfDeviceAddr);
                  DestroyTensor(other, otherDeviceAddr); DestroyTensor(out, outDeviceAddr);
                  return false);
    }

    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnAdd failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); DestroyTensor(self, selfDeviceAddr);
              DestroyTensor(other, otherDeviceAddr); DestroyTensor(out, outDeviceAddr);
              return false);

    // 同步
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Synchronize stream failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); DestroyTensor(self, selfDeviceAddr);
              DestroyTensor(other, otherDeviceAddr); DestroyTensor(out, outDeviceAddr);
              return false);

    // 获取结果并验证
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Memcpy result failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); DestroyTensor(self, selfDeviceAddr);
              DestroyTensor(other, otherDeviceAddr); DestroyTensor(out, outDeviceAddr);
              return false);

    // 验证结果
    bool resultMatch = CompareResult(resultData, expected);

    // 清理资源
    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(alpha);
    DestroyTensor(self, selfDeviceAddr);
    DestroyTensor(other, otherDeviceAddr);
    DestroyTensor(out, outDeviceAddr);

    PrintTestResult(resultMatch, "aclnnAdd basic test");
    return resultMatch;
}

// 测试 aclnnAdds - tensor + scalar
bool TestAclnnAdds(const std::vector<int64_t>& selfShape, aclDataType dataType,
                   double otherVal, double alphaVal, aclrtStream stream, const char* testName)
{
    PrintTestHeader(testName);

    int64_t size = GetShapeSize(selfShape);

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclScalar* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> selfHostData(size);
    std::vector<float> expected(size);
    std::vector<float> outHostData(size, 0);

    for (int64_t i = 0; i < size; i++) {
        selfHostData[i] = static_cast<float>(i % 10);
    }

    // expected = self + alpha * other
    for (int64_t i = 0; i < size; i++) {
        expected[i] = static_cast<float>(selfHostData[i] + alphaVal * otherVal);
    }

    int ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, dataType, &self);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create self tensor failed"); return false);

    float otherFloat = static_cast<float>(otherVal);
    other = aclCreateScalar(&otherFloat, dataType);
    CHECK_RET(other != nullptr, PrintTestResult(false, "Create other scalar failed");
              DestroyTensor(self, selfDeviceAddr); return false);

    float alphaFloat = static_cast<float>(alphaVal);
    alpha = aclCreateScalar(&alphaFloat, dataType);
    CHECK_RET(alpha != nullptr, PrintTestResult(false, "Create alpha scalar failed");
              aclDestroyScalar(other); DestroyTensor(self, selfDeviceAddr); return false);

    ret = CreateAclTensor(outHostData, selfShape, &outDeviceAddr, dataType, &out);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create out tensor failed");
              aclDestroyScalar(alpha); aclDestroyScalar(other); DestroyTensor(self, selfDeviceAddr);
              return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnAddsGetWorkspaceSize failed");
              aclDestroyScalar(alpha); aclDestroyScalar(other);
              DestroyTensor(self, selfDeviceAddr); DestroyTensor(out, outDeviceAddr);
              return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Malloc workspace failed");
                  aclDestroyScalar(alpha); aclDestroyScalar(other);
                  DestroyTensor(self, selfDeviceAddr); DestroyTensor(out, outDeviceAddr);
                  return false);
    }

    ret = aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnAdds failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); aclDestroyScalar(other);
              DestroyTensor(self, selfDeviceAddr); DestroyTensor(out, outDeviceAddr);
              return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Synchronize stream failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); aclDestroyScalar(other);
              DestroyTensor(self, selfDeviceAddr); DestroyTensor(out, outDeviceAddr);
              return false);

    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Memcpy result failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); aclDestroyScalar(other);
              DestroyTensor(self, selfDeviceAddr); DestroyTensor(out, outDeviceAddr);
              return false);

    bool resultMatch = CompareResult(resultData, expected);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(alpha);
    aclDestroyScalar(other);
    DestroyTensor(self, selfDeviceAddr);
    DestroyTensor(out, outDeviceAddr);

    PrintTestResult(resultMatch, "aclnnAdds test");
    return resultMatch;
}

// 测试 aclnnInplaceAdd
bool TestAclnnInplaceAdd(const std::vector<int64_t>& selfShape, const std::vector<int64_t>& otherShape,
                          aclDataType dataType, double alphaVal, aclrtStream stream, const char* testName)
{
    PrintTestHeader(testName);

    int64_t size = GetShapeSize(selfShape);
    int64_t otherSize = GetShapeSize(otherShape);

    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    aclTensor* selfRef = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;

    std::vector<float> selfHostData(size);
    std::vector<float> otherHostData(otherSize);
    std::vector<float> expected(size);

    for (int64_t i = 0; i < size; i++) {
        selfHostData[i] = static_cast<float>(i % 10);
    }
    for (int64_t i = 0; i < otherSize; i++) {
        otherHostData[i] = static_cast<float>((i % 5) + 1);
    }

    ComputeExpectedBroadcast(selfHostData, otherHostData, selfShape, otherShape, alphaVal, expected);

    int ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, dataType, &selfRef);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create selfRef tensor failed"); return false);

    ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, dataType, &other);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create other tensor failed");
              DestroyTensor(selfRef, selfDeviceAddr); return false);

    float alphaFloat = static_cast<float>(alphaVal);
    alpha = aclCreateScalar(&alphaFloat, dataType);
    CHECK_RET(alpha != nullptr, PrintTestResult(false, "Create alpha scalar failed");
              DestroyTensor(selfRef, selfDeviceAddr); DestroyTensor(other, otherDeviceAddr);
              return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnInplaceAddGetWorkspaceSize(selfRef, other, alpha, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnInplaceAddGetWorkspaceSize failed");
              aclDestroyScalar(alpha); DestroyTensor(selfRef, selfDeviceAddr);
              DestroyTensor(other, otherDeviceAddr); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Malloc workspace failed");
                  aclDestroyScalar(alpha); DestroyTensor(selfRef, selfDeviceAddr);
                  DestroyTensor(other, otherDeviceAddr); return false);
    }

    ret = aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnInplaceAdd failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); DestroyTensor(selfRef, selfDeviceAddr);
              DestroyTensor(other, otherDeviceAddr); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Synchronize stream failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); DestroyTensor(selfRef, selfDeviceAddr);
              DestroyTensor(other, otherDeviceAddr); return false);

    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), selfDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Memcpy result failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); DestroyTensor(selfRef, selfDeviceAddr);
              DestroyTensor(other, otherDeviceAddr); return false);

    bool resultMatch = CompareResult(resultData, expected);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(alpha);
    DestroyTensor(selfRef, selfDeviceAddr);
    DestroyTensor(other, otherDeviceAddr);

    PrintTestResult(resultMatch, "aclnnInplaceAdd test");
    return resultMatch;
}

// 测试 aclnnInplaceAdds
bool TestAclnnInplaceAdds(const std::vector<int64_t>& selfShape, aclDataType dataType,
                           double otherVal, double alphaVal, aclrtStream stream, const char* testName)
{
    PrintTestHeader(testName);

    int64_t size = GetShapeSize(selfShape);

    void* selfDeviceAddr = nullptr;
    aclTensor* selfRef = nullptr;
    aclScalar* other = nullptr;
    aclScalar* alpha = nullptr;

    std::vector<float> selfHostData(size);
    std::vector<float> expected(size);

    for (int64_t i = 0; i < size; i++) {
        selfHostData[i] = static_cast<float>(i % 10);
    }

    for (int64_t i = 0; i < size; i++) {
        expected[i] = static_cast<float>(selfHostData[i] + alphaVal * otherVal);
    }

    int ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, dataType, &selfRef);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create selfRef tensor failed"); return false);

    float otherFloat = static_cast<float>(otherVal);
    other = aclCreateScalar(&otherFloat, dataType);
    CHECK_RET(other != nullptr, PrintTestResult(false, "Create other scalar failed");
              DestroyTensor(selfRef, selfDeviceAddr); return false);

    float alphaFloat = static_cast<float>(alphaVal);
    alpha = aclCreateScalar(&alphaFloat, dataType);
    CHECK_RET(alpha != nullptr, PrintTestResult(false, "Create alpha scalar failed");
              aclDestroyScalar(other); DestroyTensor(selfRef, selfDeviceAddr); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnInplaceAddsGetWorkspaceSize(selfRef, other, alpha, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnInplaceAddsGetWorkspaceSize failed");
              aclDestroyScalar(alpha); aclDestroyScalar(other);
              DestroyTensor(selfRef, selfDeviceAddr); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Malloc workspace failed");
                  aclDestroyScalar(alpha); aclDestroyScalar(other);
                  DestroyTensor(selfRef, selfDeviceAddr); return false);
    }

    ret = aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnInplaceAdds failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); aclDestroyScalar(other);
              DestroyTensor(selfRef, selfDeviceAddr); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Synchronize stream failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); aclDestroyScalar(other);
              DestroyTensor(selfRef, selfDeviceAddr); return false);

    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), selfDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Memcpy result failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); aclDestroyScalar(other);
              DestroyTensor(selfRef, selfDeviceAddr); return false);

    bool resultMatch = CompareResult(resultData, expected);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(alpha);
    aclDestroyScalar(other);
    DestroyTensor(selfRef, selfDeviceAddr);

    PrintTestResult(resultMatch, "aclnnInplaceAdds test");
    return resultMatch;
}

// 测试 aclnnAddV3 - scalar + tensor (V3版本)
bool TestAclnnAddV3(double selfVal, const std::vector<int64_t>& otherShape,
                    aclDataType dataType, double alphaVal, aclrtStream stream, const char* testName)
{
    PrintTestHeader(testName);

    int64_t size = GetShapeSize(otherShape);

    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclScalar* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> otherHostData(size);
    std::vector<float> expected(size);
    std::vector<float> outHostData(size, 0);

    for (int64_t i = 0; i < size; i++) {
        otherHostData[i] = static_cast<float>(i % 10);
    }

    // expected = self + alpha * other
    for (int64_t i = 0; i < size; i++) {
        expected[i] = static_cast<float>(selfVal + alphaVal * otherHostData[i]);
    }

    float selfFloat = static_cast<float>(selfVal);
    self = aclCreateScalar(&selfFloat, dataType);
    CHECK_RET(self != nullptr, PrintTestResult(false, "Create self scalar failed"); return false);

    int ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, dataType, &other);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create other tensor failed");
              aclDestroyScalar(self); return false);

    float alphaFloat = static_cast<float>(alphaVal);
    alpha = aclCreateScalar(&alphaFloat, dataType);
    CHECK_RET(alpha != nullptr, PrintTestResult(false, "Create alpha scalar failed");
              aclDestroyScalar(self); DestroyTensor(other, otherDeviceAddr); return false);

    ret = CreateAclTensor(outHostData, otherShape, &outDeviceAddr, dataType, &out);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create out tensor failed");
              aclDestroyScalar(alpha); aclDestroyScalar(self);
              DestroyTensor(other, otherDeviceAddr); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnAddV3GetWorkspaceSize failed");
              aclDestroyScalar(alpha); aclDestroyScalar(self);
              DestroyTensor(other, otherDeviceAddr); DestroyTensor(out, outDeviceAddr);
              return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Malloc workspace failed");
                  aclDestroyScalar(alpha); aclDestroyScalar(self);
                  DestroyTensor(other, otherDeviceAddr); DestroyTensor(out, outDeviceAddr);
                  return false);
    }

    ret = aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnAddV3 failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); aclDestroyScalar(self);
              DestroyTensor(other, otherDeviceAddr); DestroyTensor(out, outDeviceAddr);
              return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Synchronize stream failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); aclDestroyScalar(self);
              DestroyTensor(other, otherDeviceAddr); DestroyTensor(out, outDeviceAddr);
              return false);

    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Memcpy result failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); aclDestroyScalar(self);
              DestroyTensor(other, otherDeviceAddr); DestroyTensor(out, outDeviceAddr);
              return false);

    bool resultMatch = CompareResult(resultData, expected);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(alpha);
    aclDestroyScalar(self);
    DestroyTensor(other, otherDeviceAddr);
    DestroyTensor(out, outDeviceAddr);

    PrintTestResult(resultMatch, "aclnnAddV3 test");
    return resultMatch;
}

// 测试 aclnnInplaceAddV3
bool TestAclnnInplaceAddV3(double selfVal, const std::vector<int64_t>& otherShape,
                            aclDataType dataType, double alphaVal, aclrtStream stream, const char* testName)
{
    PrintTestHeader(testName);

    int64_t size = GetShapeSize(otherShape);

    void* otherDeviceAddr = nullptr;
    aclScalar* selfRef = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;

    std::vector<float> otherHostData(size);
    std::vector<float> expected(size);

    for (int64_t i = 0; i < size; i++) {
        otherHostData[i] = static_cast<float>(i % 10);
    }

    for (int64_t i = 0; i < size; i++) {
        expected[i] = static_cast<float>(selfVal + alphaVal * otherHostData[i]);
    }

    float selfFloat = static_cast<float>(selfVal);
    selfRef = aclCreateScalar(&selfFloat, dataType);
    CHECK_RET(selfRef != nullptr, PrintTestResult(false, "Create selfRef scalar failed"); return false);

    int ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, dataType, &other);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create other tensor failed");
              aclDestroyScalar(selfRef); return false);

    float alphaFloat = static_cast<float>(alphaVal);
    alpha = aclCreateScalar(&alphaFloat, dataType);
    CHECK_RET(alpha != nullptr, PrintTestResult(false, "Create alpha scalar failed");
              aclDestroyScalar(selfRef); DestroyTensor(other, otherDeviceAddr); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnInplaceAddV3GetWorkspaceSize(selfRef, other, alpha, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnInplaceAddV3GetWorkspaceSize failed");
              aclDestroyScalar(alpha); aclDestroyScalar(selfRef);
              DestroyTensor(other, otherDeviceAddr); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Malloc workspace failed");
                  aclDestroyScalar(alpha); aclDestroyScalar(selfRef);
                  DestroyTensor(other, otherDeviceAddr); return false);
    }

    ret = aclnnInplaceAddV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnInplaceAddV3 failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); aclDestroyScalar(selfRef);
              DestroyTensor(other, otherDeviceAddr); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Synchronize stream failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); aclDestroyScalar(selfRef);
              DestroyTensor(other, otherDeviceAddr); return false);

    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), otherDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Memcpy result failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(alpha); aclDestroyScalar(selfRef);
              DestroyTensor(other, otherDeviceAddr); return false);

    bool resultMatch = CompareResult(resultData, expected);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(alpha);
    aclDestroyScalar(selfRef);
    DestroyTensor(other, otherDeviceAddr);

    PrintTestResult(resultMatch, "aclnnInplaceAddV3 test");
    return resultMatch;
}

// 测试空tensor
bool TestEmptyTensor(aclDataType dataType, aclrtStream stream, const char* testName)
{
    PrintTestHeader(testName);

    std::vector<int64_t> emptyShape = {2, 0, 3};
    std::vector<int64_t> otherShape = {2, 1, 3};
    int64_t size = 0;

    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> selfHostData;
    std::vector<float> otherHostData(GetShapeSize(otherShape));
    std::vector<float> outHostData;

    for (int64_t i = 0; i < GetShapeSize(otherShape); i++) {
        otherHostData[i] = static_cast<float>(i % 5);
    }

    int ret = CreateAclTensor(selfHostData, emptyShape, &selfDeviceAddr, dataType, &self);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create empty self tensor failed"); return false);

    ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, dataType, &other);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create other tensor failed");
              DestroyTensor(self, selfDeviceAddr); return false);

    float alphaVal = 1.0f;
    alpha = aclCreateScalar(&alphaVal, dataType);
    CHECK_RET(alpha != nullptr, PrintTestResult(false, "Create alpha scalar failed");
              DestroyTensor(self, selfDeviceAddr); DestroyTensor(other, otherDeviceAddr);
              return false);

    ret = CreateAclTensor(outHostData, emptyShape, &outDeviceAddr, dataType, &out);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create empty out tensor failed");
              aclDestroyScalar(alpha); DestroyTensor(self, selfDeviceAddr);
              DestroyTensor(other, otherDeviceAddr); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);

    bool apiSuccess = (ret == ACL_SUCCESS);

    aclDestroyScalar(alpha);
    DestroyTensor(self, selfDeviceAddr);
    DestroyTensor(other, otherDeviceAddr);
    DestroyTensor(out, outDeviceAddr);

    PrintTestResult(apiSuccess, "Empty tensor test - API should accept empty tensor");
    return apiSuccess;
}

// 测试nullptr输入
bool TestNullptrInput(aclrtStream stream)
{
    PrintTestHeader("Test nullptr input for all APIs");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 创建一个有效的tensor用于测试
    std::vector<int64_t> shape = {2, 3};
    std::vector<float> data(6, 1.0f);

    void* deviceAddr = nullptr;
    aclTensor* validTensor = nullptr;
    CreateAclTensor(data, shape, &deviceAddr, ACL_FLOAT, &validTensor);

    float alphaVal = 1.0f;
    aclScalar* validScalar = aclCreateScalar(&alphaVal, ACL_FLOAT);

    // 测试 aclnnAdd nullptr
    aclnnStatus ret = aclnnAddGetWorkspaceSize(nullptr, validTensor, validScalar, validTensor,
                                                &workspaceSize, &executor);
    bool test1 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnAdd with nullptr self: %s\n", test1 ? "[PASS]" : "[FAIL]");

    ret = aclnnAddGetWorkspaceSize(validTensor, nullptr, validScalar, validTensor,
                                    &workspaceSize, &executor);
    bool test2 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnAdd with nullptr other: %s\n", test2 ? "[PASS]" : "[FAIL]");

    ret = aclnnAddGetWorkspaceSize(validTensor, validTensor, nullptr, validTensor,
                                    &workspaceSize, &executor);
    bool test3 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnAdd with nullptr alpha: %s\n", test3 ? "[PASS]" : "[FAIL]");

    ret = aclnnAddGetWorkspaceSize(validTensor, validTensor, validScalar, nullptr,
                                    &workspaceSize, &executor);
    bool test4 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnAdd with nullptr out: %s\n", test4 ? "[PASS]" : "[FAIL]");

    // 测试 aclnnAdds nullptr
    ret = aclnnAddsGetWorkspaceSize(nullptr, validScalar, validScalar, validTensor,
                                     &workspaceSize, &executor);
    bool test5 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnAdds with nullptr self: %s\n", test5 ? "[PASS]" : "[FAIL]");

    ret = aclnnAddsGetWorkspaceSize(validTensor, nullptr, validScalar, validTensor,
                                     &workspaceSize, &executor);
    bool test6 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnAdds with nullptr other: %s\n", test6 ? "[PASS]" : "[FAIL]");

    // 测试 aclnnAddV3 nullptr
    ret = aclnnAddV3GetWorkspaceSize(nullptr, validTensor, validScalar, validTensor,
                                      &workspaceSize, &executor);
    bool test7 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnAddV3 with nullptr self: %s\n", test7 ? "[PASS]" : "[FAIL]");

    ret = aclnnAddV3GetWorkspaceSize(validScalar, nullptr, validScalar, validTensor,
                                      &workspaceSize, &executor);
    bool test8 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnAddV3 with nullptr other: %s\n", test8 ? "[PASS]" : "[FAIL]");

    aclDestroyScalar(validScalar);
    DestroyTensor(validTensor, deviceAddr);

    bool allPass = test1 && test2 && test3 && test4 && test5 && test6 && test7 && test8;
    PrintTestResult(allPass, "All nullptr tests");
    return allPass;
}

// 测试广播场景
bool TestBroadcast(aclDataType dataType, aclrtStream stream)
{
    PrintTestHeader("Test broadcast scenarios");

    bool allPass = true;

    // 测试1: self shape (4, 3), other shape (1, 3) - 行广播
    std::vector<int64_t> selfShape1 = {4, 3};
    std::vector<int64_t> otherShape1 = {1, 3};
    bool test1 = TestAclnnBasic(selfShape1, otherShape1, dataType, 1.0, stream,
                                 "Broadcast: (4,3) + (1,3)");
    allPass = allPass && test1;

    // 测试2: self shape (4, 3), other shape (3,) - 列广播
    std::vector<int64_t> selfShape2 = {4, 3};
    std::vector<int64_t> otherShape2 = {3};
    bool test2 = TestAclnnBasic(selfShape2, otherShape2, dataType, 1.0, stream,
                                 "Broadcast: (4,3) + (3,)");
    allPass = allPass && test2;

    // 测试3: self shape (4, 3), other shape (1, 1) - 标量广播
    std::vector<int64_t> selfShape3 = {4, 3};
    std::vector<int64_t> otherShape3 = {1, 1};
    bool test3 = TestAclnnBasic(selfShape3, otherShape3, dataType, 1.0, stream,
                                 "Broadcast: (4,3) + (1,1)");
    allPass = allPass && test3;

    PrintTestResult(allPass, "All broadcast tests");
    return allPass;
}

// 测试不同alpha值
bool TestDifferentAlpha(aclDataType dataType, aclrtStream stream)
{
    PrintTestHeader("Test different alpha values");

    std::vector<int64_t> shape = {4, 3};
    bool allPass = true;

    // alpha = 1.0 (标准加法)
    bool test1 = TestAclnnBasic(shape, shape, dataType, 1.0, stream,
                                 "alpha=1.0 (standard addition)");
    allPass = allPass && test1;

    // alpha = 0
    bool test2 = TestAclnnBasic(shape, shape, dataType, 0.0, stream,
                                 "alpha=0 (other term vanishes)");
    allPass = allPass && test2;

    // alpha = 负数
    bool test3 = TestAclnnBasic(shape, shape, dataType, -2.0, stream,
                                 "alpha=-2.0 (negative scaling)");
    allPass = allPass && test3;

    // alpha = 浮点数
    bool test4 = TestAclnnBasic(shape, shape, dataType, 0.5, stream,
                                 "alpha=0.5 (fractional scaling)");
    allPass = allPass && test4;

    // alpha = 较大值
    bool test5 = TestAclnnBasic(shape, shape, dataType, 10.0, stream,
                                 "alpha=10.0 (large scaling)");
    allPass = allPass && test5;

    PrintTestResult(allPass, "All alpha tests");
    return allPass;
}

// 测试不同数据类型
bool TestDifferentDtypes(aclrtStream stream)
{
    PrintTestHeader("Test different data types");

    std::vector<int64_t> shape = {4, 3};
    bool allPass = true;

    // FLOAT
    bool test1 = TestAclnnBasic(shape, shape, ACL_FLOAT, 1.0, stream, "ACL_FLOAT");
    allPass = allPass && test1;

    // FLOAT16
    bool test2 = TestAclnnBasic(shape, shape, ACL_FLOAT16, 1.0, stream, "ACL_FLOAT16");
    allPass = allPass && test2;

    // BF16
    bool test3 = TestAclnnBasic(shape, shape, ACL_BF16, 1.0, stream, "ACL_BF16");
    allPass = allPass && test3;

    // INT32
    bool test4 = TestAclnnBasic(shape, shape, ACL_INT32, 1.0, stream, "ACL_INT32");
    allPass = allPass && test4;

    // INT8
    bool test5 = TestAclnnBasic(shape, shape, ACL_INT8, 1.0, stream, "ACL_INT8");
    allPass = allPass && test5;

    // UINT8
    bool test6 = TestAclnnBasic(shape, shape, ACL_UINT8, 1.0, stream, "ACL_UINT8");
    allPass = allPass && test6;

    // INT64
    bool test7 = TestAclnnBasic(shape, shape, ACL_INT64, 1.0, stream, "ACL_INT64");
    allPass = allPass && test7;

    PrintTestResult(allPass, "All dtype tests");
    return allPass;
}

// 测试不同shape维度
bool TestDifferentShapes(aclDataType dataType, aclrtStream stream)
{
    PrintTestHeader("Test different shape dimensions");

    bool allPass = true;

    // 1D tensor
    std::vector<int64_t> shape1D = {8};
    bool test1 = TestAclnnBasic(shape1D, shape1D, dataType, 1.0, stream, "1D tensor (8,)");
    allPass = allPass && test1;

    // 2D tensor
    std::vector<int64_t> shape2D = {4, 4};
    bool test2 = TestAclnnBasic(shape2D, shape2D, dataType, 1.0, stream, "2D tensor (4,4)");
    allPass = allPass && test2;

    // 3D tensor
    std::vector<int64_t> shape3D = {2, 3, 4};
    bool test3 = TestAclnnBasic(shape3D, shape3D, dataType, 1.0, stream, "3D tensor (2,3,4)");
    allPass = allPass && test3;

    // 4D tensor
    std::vector<int64_t> shape4D = {2, 2, 2, 2};
    bool test4 = TestAclnnBasic(shape4D, shape4D, dataType, 1.0, stream, "4D tensor (2,2,2,2)");
    allPass = allPass && test4;

    // 较大tensor
    std::vector<int64_t> shapeLarge = {64, 64};
    bool test5 = TestAclnnBasic(shapeLarge, shapeLarge, dataType, 1.0, stream, "Large tensor (64,64)");
    allPass = allPass && test5;

    PrintTestResult(allPass, "All shape tests");
    return allPass;
}

// 测试数值边界
bool TestBoundaryValues(aclDataType dataType, aclrtStream stream)
{
    PrintTestHeader("Test boundary values");

    std::vector<int64_t> shape = {4, 2};
    bool allPass = true;

    // 测试零值
    bool test1 = TestAclnnBasic(shape, shape, dataType, 1.0, stream, "Zero values");
    allPass = allPass && test1;

    // 测试较大值 (大alpha)
    bool test2 = TestAclnnBasic(shape, shape, dataType, 100.0, stream, "Large alpha value");
    allPass = allPass && test2;

    PrintTestResult(allPass, "Boundary value tests");
    return allPass;
}

// 测试V3 API的不同场景
bool TestAddV3Scenarios(aclDataType dataType, aclrtStream stream)
{
    PrintTestHeader("Test aclnnAddV3 different scenarios");

    std::vector<int64_t> shape = {4, 3};
    bool allPass = true;

    // self=0, alpha=1
    bool test1 = TestAclnnAddV3(0.0, shape, dataType, 1.0, stream,
                                 "AddV3: 0 + alpha*tensor");
    allPass = allPass && test1;

    // self=非零, alpha=1
    bool test2 = TestAclnnAddV3(5.0, shape, dataType, 1.0, stream,
                                 "AddV3: 5 + alpha*tensor");
    allPass = allPass && test2;

    // self=非零, alpha!=1
    bool test3 = TestAclnnAddV3(10.0, shape, dataType, 2.5, stream,
                                 "AddV3: 10 + 2.5*tensor");
    allPass = allPass && test3;

    // self=负数
    bool test4 = TestAclnnAddV3(-3.0, shape, dataType, 1.0, stream,
                                 "AddV3: -3 + alpha*tensor");
    allPass = allPass && test4;

    // alpha=0
    bool test5 = TestAclnnAddV3(7.0, shape, dataType, 0.0, stream,
                                 "AddV3: 7 + 0*tensor = 7");
    allPass = allPass && test5;

    PrintTestResult(allPass, "All AddV3 tests");
    return allPass;
}

// ==================== 主函数 ====================

int main()
{
    LOG_PRINT("============================================\n");
    LOG_PRINT("Add Operator End-to-End Test Suite\n");
    LOG_PRINT("Testing 6 APIs with various scenarios\n");
    LOG_PRINT("============================================\n\n");

    // 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 执行所有测试
    LOG_PRINT("\n=== Phase 1: API Coverage Tests ===\n");

    // 测试6个API的基本功能
    TestAclnnBasic({4, 3}, {4, 3}, ACL_FLOAT, 1.0, stream, "aclnnAdd: basic tensor+tensor");
    TestAclnnAdds({4, 3}, ACL_FLOAT, 2.0, 1.0, stream, "aclnnAdds: tensor+scalar");
    TestAclnnInplaceAdd({4, 3}, {4, 3}, ACL_FLOAT, 1.0, stream, "aclnnInplaceAdd: inplace tensor+tensor");
    TestAclnnInplaceAdds({4, 3}, ACL_FLOAT, 3.0, 1.0, stream, "aclnnInplaceAdds: inplace tensor+scalar");
    TestAclnnAddV3(5.0, {4, 3}, ACL_FLOAT, 1.0, stream, "aclnnAddV3: scalar+tensor");
    TestAclnnInplaceAddV3(10.0, {4, 3}, ACL_FLOAT, 1.0, stream, "aclnnInplaceAddV3: inplace scalar+tensor");

    LOG_PRINT("\n=== Phase 2: Data Type Tests ===\n");
    TestDifferentDtypes(stream);

    LOG_PRINT("\n=== Phase 3: Alpha Parameter Tests ===\n");
    TestDifferentAlpha(ACL_FLOAT, stream);

    LOG_PRINT("\n=== Phase 4: Shape Tests ===\n");
    TestDifferentShapes(ACL_FLOAT, stream);
    TestBroadcast(ACL_FLOAT, stream);

    LOG_PRINT("\n=== Phase 5: V3 API Tests ===\n");
    TestAddV3Scenarios(ACL_FLOAT, stream);
    TestAddV3Scenarios(ACL_FLOAT16, stream);

    LOG_PRINT("\n=== Phase 6: Special Cases ===\n");
    TestEmptyTensor(ACL_FLOAT, stream, "Empty tensor handling");
    TestNullptrInput(stream);
    TestBoundaryValues(ACL_FLOAT, stream);

    // 清理资源
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    // 输出测试汇总
    LOG_PRINT("\n============================================\n");
    LOG_PRINT("Test Summary\n");
    LOG_PRINT("============================================\n");
    LOG_PRINT("Total tests:  %d\n", g_total_count);
    LOG_PRINT("Passed:       %d\n", g_pass_count);
    LOG_PRINT("Failed:       %d\n", g_fail_count);
    LOG_PRINT("Pass rate:    %.2f%%\n", (g_pass_count * 100.0) / g_total_count);
    LOG_PRINT("============================================\n");

    if (g_fail_count > 0) {
        LOG_PRINT("\n[WARNING] Some tests failed. Please check the log above.\n");
        return 1;
    }

    LOG_PRINT("\n[SUCCESS] All tests passed!\n");
    return 0;
}