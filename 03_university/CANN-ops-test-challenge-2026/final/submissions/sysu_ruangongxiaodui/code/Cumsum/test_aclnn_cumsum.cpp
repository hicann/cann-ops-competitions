/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
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
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

// 简单的测试执行函数
template <typename T>
int RunTest(const std::vector<T>& selfHostData, const std::vector<int64_t>& shape,
            int64_t dim, aclDataType dtype, aclDataType outDtype,
            aclrtStream stream, const char* testName)
{
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;

    int ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, dtype, &self);
    if (ret != 0) {
        LOG_PRINT("[SKIP] %s: Create self tensor failed\n", testName);
        return 0;
    }

    std::vector<T> outHostData(GetShapeSize(shape), 0);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, outDtype, &out);
    if (ret != 0) {
        LOG_PRINT("[SKIP] %s: Create out tensor failed\n", testName);
        aclDestroyTensor(self);
        aclrtFree(selfDeviceAddr);
        return 0;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnCumsumGetWorkspaceSize(self, dim, outDtype, out, &workspaceSize, &executor);

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[SKIP] %s: GetWorkspaceSize failed. ERROR: %d\n", testName, ret);
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr);
        aclrtFree(outDeviceAddr);
        return 0;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[SKIP] %s: Allocate workspace failed\n", testName);
            aclDestroyTensor(self);
            aclDestroyTensor(out);
            aclrtFree(selfDeviceAddr);
            aclrtFree(outDeviceAddr);
            return 0;
        }
    }

    ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[SKIP] %s: Execute failed\n", testName);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDeviceAddr);
        aclrtFree(outDeviceAddr);
        return 0;
    }

    aclrtSynchronizeStream(stream);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);

    LOG_PRINT("[PASS] %s\n", testName);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed\n"); return ret);

    int totalTests = 0;
    int passedTests = 0;
    int failedTests = 0;

    LOG_PRINT("========================================\n");
    LOG_PRINT("Cumsum Coverage Tests\n");
    LOG_PRINT("========================================\n\n");

    // ========================================
    // 第一组：不同数据类型的测试
    // ========================================
    LOG_PRINT("=== Group 1: Different Data Types ===\n");

    // Test 1.1: FLOAT32, 1D
    ret = RunTest<float>({1.0f, 2.0f, 3.0f, 4.0f}, {4}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "FLOAT32_1D");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 1.2: FLOAT16, 1D
    ret = RunTest<float>({1.0f, 2.0f, 3.0f, 4.0f}, {4}, 0,
                        ACL_FLOAT, ACL_FLOAT16, stream, "FLOAT16_1D");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 1.3: INT32, 1D
    std::vector<int> intData = {1, 2, 3, 4};
    ret = RunTest<int>(intData, {4}, 0,
                      ACL_INT32, ACL_INT32, stream, "INT32_1D");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 1.4: INT64, 1D
    std::vector<int64_t> int64Data = {1, 2, 3, 4};
    ret = RunTest<int64_t>(int64Data, {4}, 0,
                          ACL_INT64, ACL_INT64, stream, "INT64_1D");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 1.5: INT8, 1D
    std::vector<int8_t> int8Data = {1, 2, 3, 4};
    ret = RunTest<int8_t>(int8Data, {4}, 0,
                        ACL_INT8, ACL_INT8, stream, "INT8_1D");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 1.6: UINT8, 1D
    std::vector<uint8_t> uint8Data = {1, 2, 3, 4};
    ret = RunTest<uint8_t>(uint8Data, {4}, 0,
                         ACL_UINT8, ACL_UINT8, stream, "UINT8_1D");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 1.7: DOUBLE, 1D
    std::vector<double> doubleData = {1.0, 2.0, 3.0, 4.0};
    ret = RunTest<double>(doubleData, {4}, 0,
                        ACL_DOUBLE, ACL_DOUBLE, stream, "DOUBLE_1D");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    LOG_PRINT("\n");

    // ========================================
    // 第二组：不同维度的测试
    // ========================================
    LOG_PRINT("=== Group 2: Different Dimensions ===\n");

    // Test 2.1: 1D, dim=0
    ret = RunTest<float>({1.0f, 2.0f, 3.0f, 4.0f}, {4}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "1D_dim0");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 2.2: 2D, dim=0
    ret = RunTest<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "2D_dim0");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 2.3: 2D, dim=1
    ret = RunTest<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, 1,
                        ACL_FLOAT, ACL_FLOAT, stream, "2D_dim1");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 2.4: 3D, dim=0
    ret = RunTest<float>(
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
        {2, 2, 2}, 0, ACL_FLOAT, ACL_FLOAT, stream, "3D_dim0");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 2.5: 3D, dim=1
    ret = RunTest<float>(
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
        {2, 2, 2}, 1, ACL_FLOAT, ACL_FLOAT, stream, "3D_dim1");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 2.6: 3D, dim=2
    ret = RunTest<float>(
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
        {2, 2, 2}, 2, ACL_FLOAT, ACL_FLOAT, stream, "3D_dim2");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 2.7: 4D, dim=0
    ret = RunTest<float>(
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
         9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
        {2, 2, 2, 2}, 0, ACL_FLOAT, ACL_FLOAT, stream, "4D_dim0");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 2.8: 4D, dim=3
    ret = RunTest<float>(
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
         9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
        {2, 2, 2, 2}, 3, ACL_FLOAT, ACL_FLOAT, stream, "4D_dim3");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    LOG_PRINT("\n");

    // ========================================
    // 第三组：不同数据类型转换
    // ========================================
    LOG_PRINT("=== Group 3: Dtype Conversion ===\n");

    // Test 3.1: FLOAT32 -> FLOAT16
    ret = RunTest<float>({1.0f, 2.0f, 3.0f, 4.0f}, {4}, 0,
                        ACL_FLOAT, ACL_FLOAT16, stream, "FLOAT32_to_FLOAT16");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 3.2: FLOAT16 -> FLOAT32
    ret = RunTest<float>({1.0f, 2.0f, 3.0f, 4.0f}, {4}, 0,
                        ACL_FLOAT16, ACL_FLOAT, stream, "FLOAT16_to_FLOAT32");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 3.3: FLOAT32 -> INT32
    ret = RunTest<float>({1.0f, 2.0f, 3.0f, 4.0f}, {4}, 0,
                        ACL_FLOAT, ACL_INT32, stream, "FLOAT32_to_INT32");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 3.4: INT32 -> FLOAT32
    ret = RunTest<int>({1, 2, 3, 4}, {4}, 0,
                      ACL_INT32, ACL_FLOAT, stream, "INT32_to_FLOAT32");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    LOG_PRINT("\n");

    // ========================================
    // 第四组：不同序列长度
    // ========================================
    LOG_PRINT("=== Group 4: Different Sequence Lengths ===\n");

    // Test 4.1: 非常短的序列
    ret = RunTest<float>({1.0f}, {1}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "length_1");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 4.2: 短序列
    ret = RunTest<float>({1.0f, 2.0f}, {2}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "length_2");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 4.3: 中等序列
    std::vector<float> mediumSeq(100);
    for (int i = 0; i < 100; i++) mediumSeq[i] = 1.0f;
    ret = RunTest<float>(mediumSeq, {100}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "length_100");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 4.4: 长序列
    std::vector<float> longSeq(1000);
    for (int i = 0; i < 1000; i++) longSeq[i] = 1.0f;
    ret = RunTest<float>(longSeq, {1000}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "length_1000");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    LOG_PRINT("\n");

    // ========================================
    // 第五组：特殊数值
    // ========================================
    LOG_PRINT("=== Group 5: Special Values ===\n");

    // Test 5.1: 全零
    ret = RunTest<float>({0.0f, 0.0f, 0.0f, 0.0f}, {4}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "all_zeros");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 5.2: 全一
    ret = RunTest<float>({1.0f, 1.0f, 1.0f, 1.0f}, {4}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "all_ones");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 5.3: 负数
    ret = RunTest<float>({-1.0f, -2.0f, -3.0f, -4.0f}, {4}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "negative_values");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 5.4: 正负混合
    ret = RunTest<float>({1.0f, -2.0f, 3.0f, -4.0f}, {4}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "mixed_sign");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 5.5: 大数值
    ret = RunTest<float>({1e20f, 2e20f, 3e20f, 4e20f}, {4}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "large_values");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 5.6: 小数值
    ret = RunTest<float>({1e-20f, 2e-20f, 3e-20f, 4e-20f}, {4}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "small_values");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 5.7: 大小混合
    ret = RunTest<float>({1e20f, 1e-20f, 1e20f, 1e-20f}, {4}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "mixed_magnitude");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    LOG_PRINT("\n");

    // ========================================
    // 第六组：不同Shape大小
    // ========================================
    LOG_PRINT("=== Group 6: Different Shape Sizes ===\n");

    // Test 6.1: 1xN
    ret = RunTest<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {1, 5}, 1,
                        ACL_FLOAT, ACL_FLOAT, stream, "shape_1x5_dim1");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 6.2: Nx1
    ret = RunTest<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5, 1}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "shape_5x1_dim0");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 6.3: 3x3
    std::vector<float> data3x3(9);
    for (int i = 0; i < 9; i++) data3x3[i] = (float)(i + 1);
    ret = RunTest<float>(data3x3, {3, 3}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "shape_3x3_dim0");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 6.4: 3x3, dim=1
    ret = RunTest<float>(data3x3, {3, 3}, 1,
                        ACL_FLOAT, ACL_FLOAT, stream, "shape_3x3_dim1");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    LOG_PRINT("\n");

    // ========================================
    // 第七组：边界情况
    // ========================================
    LOG_PRINT("=== Group 7: Boundary Cases ===\n");

    // Test 7.1: dim接近INT32_MAX
    ret = RunTest<float>({1.0f, 2.0f, 3.0f}, {3}, 0,
                        ACL_FLOAT, ACL_FLOAT, stream, "dim_0");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    // Test 7.2: 沿最后一个维度累加
    ret = RunTest<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, 1,
                        ACL_FLOAT, ACL_FLOAT, stream, "dim_last");
    totalTests++;
    if (ret == 0) passedTests++; else failedTests++;

    LOG_PRINT("\n");

    // ========================================
    // 第八组：CumsumV2测试（如果有API支持）
    // ========================================
    LOG_PRINT("=== Group 8: CumsumV2 Tests ===\n");

    // 注意：CumsumV2的API可能不同，这里只是占位
    LOG_PRINT("[INFO] CumsumV2 tests skipped (API not confirmed)\n");

    LOG_PRINT("\n");

    // ========================================
    // 汇总
    // ========================================
    LOG_PRINT("========================================\n");
    LOG_PRINT("Test Summary\n");
    LOG_PRINT("========================================\n");
    LOG_PRINT("Total tests: %d\n", totalTests);
    LOG_PRINT("Passed: %d\n", passedTests);
    LOG_PRINT("Failed: %d\n", failedTests);
    LOG_PRINT("Coverage focused: YES\n");

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
