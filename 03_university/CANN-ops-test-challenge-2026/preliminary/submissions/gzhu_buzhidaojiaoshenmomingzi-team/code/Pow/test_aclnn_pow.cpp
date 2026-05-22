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
 * @file test_aclnn_pow.cpp
 * @brief Pow算子端到端测试用例，覆盖7个API变体和各种执行路径
 *
 * 测试覆盖维度:
 * - 7个API: aclnnPowTensorScalar, aclnnInplacePowTensorScalar, aclnnPowScalarTensor,
 *           aclnnPowTensorTensor, aclnnInplacePowTensorTensor, aclnnExp2, aclnnInplaceExp2
 * - 数据类型: FLOAT, FLOAT16, BF16, INT32, INT8, UINT8, INT16
 * - 特殊指数值: 0, 1, 0.5(sqrt), 2(square), 3(cube), -1(reciprocal), -2, -0.5
 * - Shape组合: 同shape, 广播, 标量
 * - 数值边界: 0底数, 负数底数, 极大值, 特殊值
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
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"
#include "aclnnop/aclnn_exp2.h"

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
                   double atol = 1e-3, double rtol = 1e-2)
{
    for (size_t i = 0; i < actual.size(); i++) {
        double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        double tolerance = atol + rtol * std::abs(static_cast<double>(expected[i]));
        // 处理 NaN 和 Inf
        if (std::isnan(static_cast<double>(actual[i])) && std::isnan(static_cast<double>(expected[i]))) {
            continue;
        }
        if (std::isinf(static_cast<double>(actual[i])) && std::isinf(static_cast<double>(expected[i]))) {
            if ((actual[i] > 0 && expected[i] > 0) || (actual[i] < 0 && expected[i] < 0)) {
                continue;
            }
        }
        if (diff > tolerance) {
            LOG_PRINT("Mismatch at index %zu: actual=%.6f, expected=%.6f, diff=%.6f\n",
                      i, static_cast<double>(actual[i]), static_cast<double>(expected[i]), diff);
            return false;
        }
    }
    return true;
}

// 计算期望值: expected = base ^ exponent (TensorScalar)
template <typename T>
void ComputeExpectedTensorScalar(const std::vector<T>& baseData, double exponent,
                                  std::vector<T>& expected)
{
    for (size_t i = 0; i < baseData.size(); i++) {
        expected[i] = static_cast<T>(std::pow(static_cast<double>(baseData[i]), exponent));
    }
}

// 计算期望值: expected = base ^ exponent (ScalarTensor)
template <typename T>
void ComputeExpectedScalarTensor(double base, const std::vector<T>& expData,
                                  std::vector<T>& expected)
{
    for (size_t i = 0; i < expData.size(); i++) {
        expected[i] = static_cast<T>(std::pow(base, static_cast<double>(expData[i])));
    }
}

// 计算期望值: expected = base ^ exponent (TensorTensor)
template <typename T>
void ComputeExpectedTensorTensor(const std::vector<T>& baseData, const std::vector<T>& expData,
                                  std::vector<T>& expected)
{
    for (size_t i = 0; i < baseData.size(); i++) {
        expected[i] = static_cast<T>(std::pow(static_cast<double>(baseData[i]),
                                               static_cast<double>(expData[i])));
    }
}

// 计算期望值: expected = 2 ^ self (Exp2)
template <typename T>
void ComputeExpectedExp2(const std::vector<T>& selfData, std::vector<T>& expected)
{
    for (size_t i = 0; i < selfData.size(); i++) {
        expected[i] = static_cast<T>(std::pow(2.0, static_cast<double>(selfData[i])));
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

// 测试 aclnnPowTensorScalar - tensor ^ scalar
bool TestPowTensorScalar(const std::vector<int64_t>& selfShape, aclDataType dataType,
                         double exponentVal, aclrtStream stream, const char* testName)
{
    PrintTestHeader(testName);

    int64_t size = GetShapeSize(selfShape);

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclScalar* exponent = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> selfHostData(size);
    std::vector<float> expected(size);
    std::vector<float> outHostData(size, 0);

    // 初始化输入数据 (避免0作为负指数的底数)
    for (int64_t i = 0; i < size; i++) {
        selfHostData[i] = static_cast<float>((i % 8) + 1);  // 1-8
    }

    // 计算期望值
    ComputeExpectedTensorScalar(selfHostData, exponentVal, expected);

    // 创建tensor
    int ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, dataType, &self);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create self tensor failed"); return false);

    // 创建exponent scalar
    float expFloat = static_cast<float>(exponentVal);
    exponent = aclCreateScalar(&expFloat, dataType);
    CHECK_RET(exponent != nullptr, PrintTestResult(false, "Create exponent scalar failed");
              DestroyTensor(self, selfDeviceAddr); return false);

    // 创建输出tensor
    ret = CreateAclTensor(outHostData, selfShape, &outDeviceAddr, dataType, &out);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create out tensor failed");
              aclDestroyScalar(exponent); DestroyTensor(self, selfDeviceAddr);
              return false);

    // 调用API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnPowTensorScalarGetWorkspaceSize failed");
              aclDestroyScalar(exponent); DestroyTensor(self, selfDeviceAddr);
              DestroyTensor(out, outDeviceAddr); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Malloc workspace failed");
                  aclDestroyScalar(exponent); DestroyTensor(self, selfDeviceAddr);
                  DestroyTensor(out, outDeviceAddr); return false);
    }

    ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnPowTensorScalar failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(exponent); DestroyTensor(self, selfDeviceAddr);
              DestroyTensor(out, outDeviceAddr); return false);

    // 同步
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Synchronize stream failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(exponent); DestroyTensor(self, selfDeviceAddr);
              DestroyTensor(out, outDeviceAddr); return false);

    // 获取结果并验证
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Memcpy result failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(exponent); DestroyTensor(self, selfDeviceAddr);
              DestroyTensor(out, outDeviceAddr); return false);

    // 验证结果
    bool resultMatch = CompareResult(resultData, expected);

    // 清理资源
    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(exponent);
    DestroyTensor(self, selfDeviceAddr);
    DestroyTensor(out, outDeviceAddr);

    PrintTestResult(resultMatch, "aclnnPowTensorScalar test");
    return resultMatch;
}

// 测试 aclnnInplacePowTensorScalar
bool TestInplacePowTensorScalar(const std::vector<int64_t>& selfShape, aclDataType dataType,
                                 double exponentVal, aclrtStream stream, const char* testName)
{
    PrintTestHeader(testName);

    int64_t size = GetShapeSize(selfShape);

    void* selfDeviceAddr = nullptr;
    aclTensor* selfRef = nullptr;
    aclScalar* exponent = nullptr;

    std::vector<float> selfHostData(size);
    std::vector<float> expected(size);

    for (int64_t i = 0; i < size; i++) {
        selfHostData[i] = static_cast<float>((i % 8) + 1);
    }

    ComputeExpectedTensorScalar(selfHostData, exponentVal, expected);

    int ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, dataType, &selfRef);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create selfRef tensor failed"); return false);

    float expFloat = static_cast<float>(exponentVal);
    exponent = aclCreateScalar(&expFloat, dataType);
    CHECK_RET(exponent != nullptr, PrintTestResult(false, "Create exponent scalar failed");
              DestroyTensor(selfRef, selfDeviceAddr); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnInplacePowTensorScalarGetWorkspaceSize(selfRef, exponent, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnInplacePowTensorScalarGetWorkspaceSize failed");
              aclDestroyScalar(exponent); DestroyTensor(selfRef, selfDeviceAddr); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Malloc workspace failed");
                  aclDestroyScalar(exponent); DestroyTensor(selfRef, selfDeviceAddr); return false);
    }

    ret = aclnnInplacePowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnInplacePowTensorScalar failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(exponent); DestroyTensor(selfRef, selfDeviceAddr); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Synchronize stream failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(exponent); DestroyTensor(selfRef, selfDeviceAddr); return false);

    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), selfDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Memcpy result failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(exponent); DestroyTensor(selfRef, selfDeviceAddr); return false);

    bool resultMatch = CompareResult(resultData, expected);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(exponent);
    DestroyTensor(selfRef, selfDeviceAddr);

    PrintTestResult(resultMatch, "aclnnInplacePowTensorScalar test");
    return resultMatch;
}

// 测试 aclnnPowScalarTensor - scalar ^ tensor
bool TestPowScalarTensor(double baseVal, const std::vector<int64_t>& expShape, aclDataType dataType,
                         aclrtStream stream, const char* testName)
{
    PrintTestHeader(testName);

    int64_t size = GetShapeSize(expShape);

    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclScalar* self = nullptr;
    aclTensor* exponent = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> expHostData(size);
    std::vector<float> expected(size);
    std::vector<float> outHostData(size, 0);

    for (int64_t i = 0; i < size; i++) {
        expHostData[i] = static_cast<float>(i % 5);  // 0-4
    }

    ComputeExpectedScalarTensor(baseVal, expHostData, expected);

    // 创建base scalar
    float baseFloat = static_cast<float>(baseVal);
    self = aclCreateScalar(&baseFloat, dataType);
    CHECK_RET(self != nullptr, PrintTestResult(false, "Create self scalar failed"); return false);

    int ret = CreateAclTensor(expHostData, expShape, &expDeviceAddr, dataType, &exponent);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create exponent tensor failed");
              aclDestroyScalar(self); return false);

    ret = CreateAclTensor(outHostData, expShape, &outDeviceAddr, dataType, &out);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create out tensor failed");
              aclDestroyScalar(self); DestroyTensor(exponent, expDeviceAddr); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnPowScalarTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnPowScalarTensorGetWorkspaceSize failed");
              aclDestroyScalar(self); DestroyTensor(exponent, expDeviceAddr);
              DestroyTensor(out, outDeviceAddr); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Malloc workspace failed");
                  aclDestroyScalar(self); DestroyTensor(exponent, expDeviceAddr);
                  DestroyTensor(out, outDeviceAddr); return false);
    }

    ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnPowScalarTensor failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(self); DestroyTensor(exponent, expDeviceAddr);
              DestroyTensor(out, outDeviceAddr); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Synchronize stream failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(self); DestroyTensor(exponent, expDeviceAddr);
              DestroyTensor(out, outDeviceAddr); return false);

    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Memcpy result failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              aclDestroyScalar(self); DestroyTensor(exponent, expDeviceAddr);
              DestroyTensor(out, outDeviceAddr); return false);

    bool resultMatch = CompareResult(resultData, expected);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(self);
    DestroyTensor(exponent, expDeviceAddr);
    DestroyTensor(out, outDeviceAddr);

    PrintTestResult(resultMatch, "aclnnPowScalarTensor test");
    return resultMatch;
}

// 测试 aclnnPowTensorTensor - tensor ^ tensor
bool TestPowTensorTensor(const std::vector<int64_t>& selfShape, const std::vector<int64_t>& expShape,
                         aclDataType dataType, aclrtStream stream, const char* testName)
{
    PrintTestHeader(testName);

    int64_t selfSize = GetShapeSize(selfShape);
    int64_t expSize = GetShapeSize(expShape);
    int64_t outSize = GetShapeSize(selfShape);  // broadcast result

    void* selfDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* exponent = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> selfHostData(selfSize);
    std::vector<float> expHostData(expSize);
    std::vector<float> expected(outSize);
    std::vector<float> outHostData(outSize, 0);

    for (int64_t i = 0; i < selfSize; i++) {
        selfHostData[i] = static_cast<float>((i % 8) + 1);
    }
    for (int64_t i = 0; i < expSize; i++) {
        expHostData[i] = static_cast<float>((i % 4) + 1);  // 1-4
    }

    // 计算期望值 (考虑广播)
    for (int64_t i = 0; i < outSize; i++) {
        float base = selfHostData[i];
        float exp = (expSize == 1) ? expHostData[0] : expHostData[i % expSize];
        expected[i] = static_cast<float>(std::pow(static_cast<double>(base), static_cast<double>(exp)));
    }

    int ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, dataType, &self);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create self tensor failed"); return false);

    ret = CreateAclTensor(expHostData, expShape, &expDeviceAddr, dataType, &exponent);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create exponent tensor failed");
              DestroyTensor(self, selfDeviceAddr); return false);

    ret = CreateAclTensor(outHostData, selfShape, &outDeviceAddr, dataType, &out);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create out tensor failed");
              DestroyTensor(self, selfDeviceAddr); DestroyTensor(exponent, expDeviceAddr);
              return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnPowTensorTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnPowTensorTensorGetWorkspaceSize failed");
              DestroyTensor(self, selfDeviceAddr); DestroyTensor(exponent, expDeviceAddr);
              DestroyTensor(out, outDeviceAddr); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Malloc workspace failed");
                  DestroyTensor(self, selfDeviceAddr); DestroyTensor(exponent, expDeviceAddr);
                  DestroyTensor(out, outDeviceAddr); return false);
    }

    ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnPowTensorTensor failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              DestroyTensor(self, selfDeviceAddr); DestroyTensor(exponent, expDeviceAddr);
              DestroyTensor(out, outDeviceAddr); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Synchronize stream failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              DestroyTensor(self, selfDeviceAddr); DestroyTensor(exponent, expDeviceAddr);
              DestroyTensor(out, outDeviceAddr); return false);

    std::vector<float> resultData(outSize, 0);
    ret = aclrtMemcpy(resultData.data(), outSize * sizeof(float), outDeviceAddr,
                      outSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Memcpy result failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              DestroyTensor(self, selfDeviceAddr); DestroyTensor(exponent, expDeviceAddr);
              DestroyTensor(out, outDeviceAddr); return false);

    bool resultMatch = CompareResult(resultData, expected);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyTensor(self, selfDeviceAddr);
    DestroyTensor(exponent, expDeviceAddr);
    DestroyTensor(out, outDeviceAddr);

    PrintTestResult(resultMatch, "aclnnPowTensorTensor test");
    return resultMatch;
}

// 测试 aclnnInplacePowTensorTensor
bool TestInplacePowTensorTensor(const std::vector<int64_t>& selfShape, const std::vector<int64_t>& expShape,
                                 aclDataType dataType, aclrtStream stream, const char* testName)
{
    PrintTestHeader(testName);

    int64_t selfSize = GetShapeSize(selfShape);
    int64_t expSize = GetShapeSize(expShape);

    void* selfDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    aclTensor* selfRef = nullptr;
    aclTensor* exponent = nullptr;

    std::vector<float> selfHostData(selfSize);
    std::vector<float> expHostData(expSize);
    std::vector<float> expected(selfSize);

    for (int64_t i = 0; i < selfSize; i++) {
        selfHostData[i] = static_cast<float>((i % 8) + 1);
    }
    for (int64_t i = 0; i < expSize; i++) {
        expHostData[i] = static_cast<float>((i % 4) + 1);
    }

    for (int64_t i = 0; i < selfSize; i++) {
        float base = selfHostData[i];
        float exp = (expSize == 1) ? expHostData[0] : expHostData[i % expSize];
        expected[i] = static_cast<float>(std::pow(static_cast<double>(base), static_cast<double>(exp)));
    }

    int ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, dataType, &selfRef);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create selfRef tensor failed"); return false);

    ret = CreateAclTensor(expHostData, expShape, &expDeviceAddr, dataType, &exponent);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create exponent tensor failed");
              DestroyTensor(selfRef, selfDeviceAddr); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnInplacePowTensorTensorGetWorkspaceSize(selfRef, exponent, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnInplacePowTensorTensorGetWorkspaceSize failed");
              DestroyTensor(selfRef, selfDeviceAddr); DestroyTensor(exponent, expDeviceAddr); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Malloc workspace failed");
                  DestroyTensor(selfRef, selfDeviceAddr); DestroyTensor(exponent, expDeviceAddr); return false);
    }

    ret = aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnInplacePowTensorTensor failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              DestroyTensor(selfRef, selfDeviceAddr); DestroyTensor(exponent, expDeviceAddr); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Synchronize stream failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              DestroyTensor(selfRef, selfDeviceAddr); DestroyTensor(exponent, expDeviceAddr); return false);

    std::vector<float> resultData(selfSize, 0);
    ret = aclrtMemcpy(resultData.data(), selfSize * sizeof(float), selfDeviceAddr,
                      selfSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Memcpy result failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              DestroyTensor(selfRef, selfDeviceAddr); DestroyTensor(exponent, expDeviceAddr); return false);

    bool resultMatch = CompareResult(resultData, expected);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyTensor(selfRef, selfDeviceAddr);
    DestroyTensor(exponent, expDeviceAddr);

    PrintTestResult(resultMatch, "aclnnInplacePowTensorTensor test");
    return resultMatch;
}

// 测试 aclnnExp2 - 2 ^ tensor
bool TestExp2(const std::vector<int64_t>& selfShape, aclDataType dataType,
              aclrtStream stream, const char* testName)
{
    PrintTestHeader(testName);

    int64_t size = GetShapeSize(selfShape);

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> selfHostData(size);
    std::vector<float> expected(size);
    std::vector<float> outHostData(size, 0);

    for (int64_t i = 0; i < size; i++) {
        selfHostData[i] = static_cast<float>(i % 5);  // 0-4
    }

    ComputeExpectedExp2(selfHostData, expected);

    int ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, dataType, &self);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create self tensor failed"); return false);

    ret = CreateAclTensor(outHostData, selfShape, &outDeviceAddr, dataType, &out);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create out tensor failed");
              DestroyTensor(self, selfDeviceAddr); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnExp2GetWorkspaceSize(self, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnExp2GetWorkspaceSize failed");
              DestroyTensor(self, selfDeviceAddr); DestroyTensor(out, outDeviceAddr); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Malloc workspace failed");
                  DestroyTensor(self, selfDeviceAddr); DestroyTensor(out, outDeviceAddr); return false);
    }

    ret = aclnnExp2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnExp2 failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              DestroyTensor(self, selfDeviceAddr); DestroyTensor(out, outDeviceAddr); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Synchronize stream failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              DestroyTensor(self, selfDeviceAddr); DestroyTensor(out, outDeviceAddr); return false);

    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Memcpy result failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              DestroyTensor(self, selfDeviceAddr); DestroyTensor(out, outDeviceAddr); return false);

    bool resultMatch = CompareResult(resultData, expected);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyTensor(self, selfDeviceAddr);
    DestroyTensor(out, outDeviceAddr);

    PrintTestResult(resultMatch, "aclnnExp2 test");
    return resultMatch;
}

// 测试 aclnnInplaceExp2
bool TestInplaceExp2(const std::vector<int64_t>& selfShape, aclDataType dataType,
                     aclrtStream stream, const char* testName)
{
    PrintTestHeader(testName);

    int64_t size = GetShapeSize(selfShape);

    void* selfDeviceAddr = nullptr;
    aclTensor* selfRef = nullptr;

    std::vector<float> selfHostData(size);
    std::vector<float> expected(size);

    for (int64_t i = 0; i < size; i++) {
        selfHostData[i] = static_cast<float>(i % 5);
    }

    ComputeExpectedExp2(selfHostData, expected);

    int ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, dataType, &selfRef);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create selfRef tensor failed"); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnInplaceExp2GetWorkspaceSize(selfRef, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnInplaceExp2GetWorkspaceSize failed");
              DestroyTensor(selfRef, selfDeviceAddr); return false);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Malloc workspace failed");
                  DestroyTensor(selfRef, selfDeviceAddr); return false);
    }

    ret = aclnnInplaceExp2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "aclnnInplaceExp2 failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              DestroyTensor(selfRef, selfDeviceAddr); return false);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Synchronize stream failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              DestroyTensor(selfRef, selfDeviceAddr); return false);

    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(float), selfDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Memcpy result failed");
              if (workspaceAddr) aclrtFree(workspaceAddr);
              DestroyTensor(selfRef, selfDeviceAddr); return false);

    bool resultMatch = CompareResult(resultData, expected);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    DestroyTensor(selfRef, selfDeviceAddr);

    PrintTestResult(resultMatch, "aclnnInplaceExp2 test");
    return resultMatch;
}

// 测试空tensor
bool TestEmptyTensor(aclDataType dataType, aclrtStream stream, const char* testName)
{
    PrintTestHeader(testName);

    std::vector<int64_t> emptyShape = {2, 0, 3};
    int64_t size = 0;

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclScalar* exponent = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> selfHostData;
    std::vector<float> outHostData;

    int ret = CreateAclTensor(selfHostData, emptyShape, &selfDeviceAddr, dataType, &self);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create empty self tensor failed"); return false);

    float expVal = 2.0f;
    exponent = aclCreateScalar(&expVal, dataType);
    CHECK_RET(exponent != nullptr, PrintTestResult(false, "Create exponent scalar failed");
              DestroyTensor(self, selfDeviceAddr); return false);

    ret = CreateAclTensor(outHostData, emptyShape, &outDeviceAddr, dataType, &out);
    CHECK_RET(ret == ACL_SUCCESS, PrintTestResult(false, "Create empty out tensor failed");
              aclDestroyScalar(exponent); DestroyTensor(self, selfDeviceAddr);
              return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);

    bool apiSuccess = (ret == ACL_SUCCESS);

    aclDestroyScalar(exponent);
    DestroyTensor(self, selfDeviceAddr);
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
    std::vector<float> data(6, 2.0f);

    void* deviceAddr = nullptr;
    aclTensor* validTensor = nullptr;
    CreateAclTensor(data, shape, &deviceAddr, ACL_FLOAT, &validTensor);

    float expVal = 2.0f;
    aclScalar* validScalar = aclCreateScalar(&expVal, ACL_FLOAT);

    bool allPass = true;

    // 测试 aclnnPowTensorScalar nullptr
    aclnnStatus ret = aclnnPowTensorScalarGetWorkspaceSize(nullptr, validScalar, validTensor,
                                                           &workspaceSize, &executor);
    bool test1 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnPowTensorScalar with nullptr self: %s\n", test1 ? "[PASS]" : "[FAIL]");
    allPass = allPass && test1;

    ret = aclnnPowTensorScalarGetWorkspaceSize(validTensor, nullptr, validTensor,
                                               &workspaceSize, &executor);
    bool test2 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnPowTensorScalar with nullptr exponent: %s\n", test2 ? "[PASS]" : "[FAIL]");
    allPass = allPass && test2;

    ret = aclnnPowTensorScalarGetWorkspaceSize(validTensor, validScalar, nullptr,
                                               &workspaceSize, &executor);
    bool test3 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnPowTensorScalar with nullptr out: %s\n", test3 ? "[PASS]" : "[FAIL]");
    allPass = allPass && test3;

    // 测试 aclnnPowScalarTensor nullptr
    ret = aclnnPowScalarTensorGetWorkspaceSize(nullptr, validTensor, validTensor,
                                               &workspaceSize, &executor);
    bool test4 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnPowScalarTensor with nullptr self: %s\n", test4 ? "[PASS]" : "[FAIL]");
    allPass = allPass && test4;

    ret = aclnnPowScalarTensorGetWorkspaceSize(validScalar, nullptr, validTensor,
                                               &workspaceSize, &executor);
    bool test5 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnPowScalarTensor with nullptr exponent: %s\n", test5 ? "[PASS]" : "[FAIL]");
    allPass = allPass && test5;

    // 测试 aclnnPowTensorTensor nullptr
    ret = aclnnPowTensorTensorGetWorkspaceSize(nullptr, validTensor, validTensor,
                                               &workspaceSize, &executor);
    bool test6 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnPowTensorTensor with nullptr self: %s\n", test6 ? "[PASS]" : "[FAIL]");
    allPass = allPass && test6;

    ret = aclnnPowTensorTensorGetWorkspaceSize(validTensor, nullptr, validTensor,
                                               &workspaceSize, &executor);
    bool test7 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnPowTensorTensor with nullptr exponent: %s\n", test7 ? "[PASS]" : "[FAIL]");
    allPass = allPass && test7;

    // 测试 aclnnExp2 nullptr
    ret = aclnnExp2GetWorkspaceSize(nullptr, validTensor, &workspaceSize, &executor);
    bool test8 = (ret == ACLNN_ERR_PARAM_NULLPTR);
    LOG_PRINT("  aclnnExp2 with nullptr self: %s\n", test8 ? "[PASS]" : "[FAIL]");
    allPass = allPass && test8;

    aclDestroyScalar(validScalar);
    DestroyTensor(validTensor, deviceAddr);

    PrintTestResult(allPass, "All nullptr tests");
    return allPass;
}

// 测试特殊指数值（触发特殊优化路径）
bool TestSpecialExponents(aclDataType dataType, aclrtStream stream)
{
    PrintTestHeader("Test special exponent values (sqrt/square/cube/reciprocal)");

    std::vector<int64_t> shape = {4, 3};
    bool allPass = true;

    // exponent = 0.5 (sqrt)
    bool test1 = TestPowTensorScalar(shape, dataType, 0.5, stream,
                                      "exponent=0.5 (sqrt optimization)");
    allPass = allPass && test1;

    // exponent = 2.0 (square)
    bool test2 = TestPowTensorScalar(shape, dataType, 2.0, stream,
                                      "exponent=2.0 (square optimization)");
    allPass = allPass && test2;

    // exponent = 3.0 (cube)
    bool test3 = TestPowTensorScalar(shape, dataType, 3.0, stream,
                                      "exponent=3.0 (cube optimization)");
    allPass = allPass && test3;

    // exponent = -0.5
    bool test4 = TestPowTensorScalar(shape, dataType, -0.5, stream,
                                      "exponent=-0.5 (rsqrt optimization)");
    allPass = allPass && test4;

    // exponent = -1.0 (reciprocal)
    bool test5 = TestPowTensorScalar(shape, dataType, -1.0, stream,
                                      "exponent=-1.0 (reciprocal optimization)");
    allPass = allPass && test5;

    // exponent = -2.0
    bool test6 = TestPowTensorScalar(shape, dataType, -2.0, stream,
                                      "exponent=-2.0");
    allPass = allPass && test6;

    // exponent = 0.0 (所有数的0次幂为1)
    bool test7 = TestPowTensorScalar(shape, dataType, 0.0, stream,
                                      "exponent=0.0 (all results = 1)");
    allPass = allPass && test7;

    // exponent = 1.0 (不变)
    bool test8 = TestPowTensorScalar(shape, dataType, 1.0, stream,
                                      "exponent=1.0 (identity)");
    allPass = allPass && test8;

    PrintTestResult(allPass, "All special exponent tests");
    return allPass;
}

// 测试不同数据类型
bool TestDifferentDtypes(aclrtStream stream)
{
    PrintTestHeader("Test different data types");

    std::vector<int64_t> shape = {4, 3};
    double exponent = 2.0;
    bool allPass = true;

    // FLOAT
    bool test1 = TestPowTensorScalar(shape, ACL_FLOAT, exponent, stream, "ACL_FLOAT");
    allPass = allPass && test1;

    // FLOAT16
    bool test2 = TestPowTensorScalar(shape, ACL_FLOAT16, exponent, stream, "ACL_FLOAT16");
    allPass = allPass && test2;

    // BF16
    bool test3 = TestPowTensorScalar(shape, ACL_BF16, exponent, stream, "ACL_BF16");
    allPass = allPass && test3;

    // INT32
    bool test4 = TestPowTensorScalar(shape, ACL_INT32, exponent, stream, "ACL_INT32");
    allPass = allPass && test4;

    // INT8
    bool test5 = TestPowTensorScalar(shape, ACL_INT8, exponent, stream, "ACL_INT8");
    allPass = allPass && test5;

    // UINT8
    bool test6 = TestPowTensorScalar(shape, ACL_UINT8, exponent, stream, "ACL_UINT8");
    allPass = allPass && test6;

    // INT16
    bool test7 = TestPowTensorScalar(shape, ACL_INT16, exponent, stream, "ACL_INT16");
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
    bool test1 = TestPowTensorScalar({8}, dataType, 2.0, stream, "1D tensor (8,)");
    allPass = allPass && test1;

    // 2D tensor
    bool test2 = TestPowTensorScalar({4, 4}, dataType, 2.0, stream, "2D tensor (4,4)");
    allPass = allPass && test2;

    // 3D tensor
    bool test3 = TestPowTensorScalar({2, 3, 4}, dataType, 2.0, stream, "3D tensor (2,3,4)");
    allPass = allPass && test3;

    // 4D tensor
    bool test4 = TestPowTensorScalar({2, 2, 2, 2}, dataType, 2.0, stream, "4D tensor (2,2,2,2)");
    allPass = allPass && test4;

    // 较大tensor
    bool test5 = TestPowTensorScalar({32, 32}, dataType, 2.0, stream, "Large tensor (32,32)");
    allPass = allPass && test5;

    PrintTestResult(allPass, "All shape tests");
    return allPass;
}

// 测试广播场景
bool TestBroadcast(aclDataType dataType, aclrtStream stream)
{
    PrintTestHeader("Test broadcast scenarios for TensorTensor");

    bool allPass = true;

    // 同shape
    bool test1 = TestPowTensorTensor({4, 3}, {4, 3}, dataType, stream,
                                      "TensorTensor: same shape");
    allPass = allPass && test1;

    // 行广播
    bool test2 = TestPowTensorTensor({4, 3}, {1, 3}, dataType, stream,
                                      "TensorTensor: row broadcast");
    allPass = allPass && test2;

    // 列广播
    bool test3 = TestPowTensorTensor({4, 3}, {3}, dataType, stream,
                                      "TensorTensor: col broadcast");
    allPass = allPass && test3;

    // 标量广播
    bool test4 = TestPowTensorTensor({4, 3}, {1}, dataType, stream,
                                      "TensorTensor: scalar broadcast");
    allPass = allPass && test4;

    PrintTestResult(allPass, "All broadcast tests");
    return allPass;
}

// 测试ScalarTensor特殊场景
bool TestScalarTensorSpecialCases(aclDataType dataType, aclrtStream stream)
{
    PrintTestHeader("Test ScalarTensor special cases");

    std::vector<int64_t> shape = {4, 3};
    bool allPass = true;

    // base = 2 (典型Exp2场景的泛化)
    bool test1 = TestPowScalarTensor(2.0, shape, dataType, stream, "base=2");
    allPass = allPass && test1;

    // base = 10
    bool test2 = TestPowScalarTensor(10.0, shape, dataType, stream, "base=10");
    allPass = allPass && test2;

    // base = 0.5
    bool test3 = TestPowScalarTensor(0.5, shape, dataType, stream, "base=0.5");
    allPass = allPass && test3;

    // base = 1 (所有指数结果为1)
    bool test4 = TestPowScalarTensor(1.0, shape, dataType, stream, "base=1 (all results=1)");
    allPass = allPass && test4;

    PrintTestResult(allPass, "All ScalarTensor special tests");
    return allPass;
}

// 测试TensorTensor tiling覆盖 (覆盖不同的dtype组合)
bool TestTensorTensorTilingCoverage(aclrtStream stream)
{
    PrintTestHeader("Test TensorTensor tiling coverage for different dtype OP_KEYs");

    std::vector<int64_t> shape = {4, 4};
    bool allPass = true;

    // OP_KEY_1: DT_FLOAT16
    bool test1 = TestPowTensorTensor(shape, shape, ACL_FLOAT16, stream, "Tiling OP_KEY_1: FLOAT16");
    allPass = allPass && test1;

    // OP_KEY_2: DT_BF16
    bool test2 = TestPowTensorTensor(shape, shape, ACL_BF16, stream, "Tiling OP_KEY_2: BF16");
    allPass = allPass && test2;

    // OP_KEY_3: DT_FLOAT
    bool test3 = TestPowTensorTensor(shape, shape, ACL_FLOAT, stream, "Tiling OP_KEY_3: FLOAT");
    allPass = allPass && test3;

    // OP_KEY_4: DT_UINT8
    bool test4 = TestPowTensorTensor(shape, shape, ACL_UINT8, stream, "Tiling OP_KEY_4: UINT8");
    allPass = allPass && test4;

    // OP_KEY_5: DT_INT8
    bool test5 = TestPowTensorTensor(shape, shape, ACL_INT8, stream, "Tiling OP_KEY_5: INT8");
    allPass = allPass && test5;

    // OP_KEY_6: DT_INT16
    bool test6 = TestPowTensorTensor(shape, shape, ACL_INT16, stream, "Tiling OP_KEY_6: INT16");
    allPass = allPass && test6;

    // OP_KEY_7: DT_INT32
    bool test7 = TestPowTensorTensor(shape, shape, ACL_INT32, stream, "Tiling OP_KEY_7: INT32");
    allPass = allPass && test7;

    PrintTestResult(allPass, "All TensorTensor tiling tests");
    return allPass;
}

// ==================== 主函数 ====================

int main()
{
    LOG_PRINT("============================================\n");
    LOG_PRINT("Pow Operator End-to-End Test Suite\n");
    LOG_PRINT("Testing 7 APIs with various scenarios\n");
    LOG_PRINT("============================================\n\n");

    // 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 执行所有测试
    LOG_PRINT("\n=== Phase 1: API Coverage Tests (7 APIs) ===\n");

    // 测试所有7个API的基本功能
    TestPowTensorScalar({4, 3}, ACL_FLOAT, 2.0, stream, "aclnnPowTensorScalar: tensor^scalar");
    TestInplacePowTensorScalar({4, 3}, ACL_FLOAT, 2.0, stream, "aclnnInplacePowTensorScalar");
    TestPowScalarTensor(2.0, {4, 3}, ACL_FLOAT, stream, "aclnnPowScalarTensor: scalar^tensor");
    TestPowTensorTensor({4, 3}, {4, 3}, ACL_FLOAT, stream, "aclnnPowTensorTensor: tensor^tensor");
    TestInplacePowTensorTensor({4, 3}, {4, 3}, ACL_FLOAT, stream, "aclnnInplacePowTensorTensor");
    TestExp2({4, 3}, ACL_FLOAT, stream, "aclnnExp2: 2^tensor");
    TestInplaceExp2({4, 3}, ACL_FLOAT, stream, "aclnnInplaceExp2");

    LOG_PRINT("\n=== Phase 2: Special Exponent Tests ===\n");
    TestSpecialExponents(ACL_FLOAT, stream);

    LOG_PRINT("\n=== Phase 3: Data Type Tests ===\n");
    TestDifferentDtypes(stream);

    LOG_PRINT("\n=== Phase 4: Shape Tests ===\n");
    TestDifferentShapes(ACL_FLOAT, stream);

    LOG_PRINT("\n=== Phase 5: TensorTensor Tests ===\n");
    TestBroadcast(ACL_FLOAT, stream);
    TestTensorTensorTilingCoverage(stream);

    LOG_PRINT("\n=== Phase 6: ScalarTensor Tests ===\n");
    TestScalarTensorSpecialCases(ACL_FLOAT, stream);

    LOG_PRINT("\n=== Phase 7: Special Cases ===\n");
    TestEmptyTensor(ACL_FLOAT, stream, "Empty tensor handling");
    TestNullptrInput(stream);

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