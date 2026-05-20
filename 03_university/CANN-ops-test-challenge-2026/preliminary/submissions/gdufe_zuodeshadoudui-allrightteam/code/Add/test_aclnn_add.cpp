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
#include <cstdint>
#include <limits>
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

static int g_total_tests = 0;
static int g_passed_tests = 0;
static int g_failed_tests = 0;

#define TEST_PASS()              \
    do {                         \
        g_total_tests++;         \
        g_passed_tests++;        \
        LOG_PRINT("  [PASS]\n"); \
    } while (0)
#define TEST_FAIL()              \
    do {                         \
        g_total_tests++;         \
        g_failed_tests++;        \
        LOG_PRINT("  [FAIL]\n"); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape)
        shapeSize *= i;
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

template <typename T>
int CreateEmptyAclTensor(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

template <typename T>
aclScalar* CreateAclScalar(T value, aclDataType dataType)
{
    return aclCreateScalar(&value, dataType);
}

// 验证函数
template <typename T>
bool VerifySameShape(
    const std::vector<T>& result, const std::vector<T>& selfData, const std::vector<T>& otherData, float alphaValue)
{
    for (size_t i = 0; i < result.size(); i++) {
        T expected = selfData[i] + static_cast<T>(alphaValue * otherData[i]);
        if (result[i] != expected)
            return false;
    }
    return true;
}

template <>
bool VerifySameShape<float>(
    const std::vector<float>& result, const std::vector<float>& selfData, const std::vector<float>& otherData,
    float alphaValue)
{
    for (size_t i = 0; i < result.size(); i++) {
        float expected = selfData[i] + alphaValue * otherData[i];
        float diff = std::abs(result[i] - expected);
        if (diff > 1e-5 * std::abs(expected) + 1e-5)
            return false;
    }
    return true;
}

// ============== aclnnAdd 测试 ==============
template <typename T>
void RunAddTest(
    const std::vector<T>& selfData, const std::vector<int64_t>& selfShape, const std::vector<T>& otherData,
    const std::vector<int64_t>& otherShape, float alphaValue, aclDataType dataType, aclrtStream stream,
    const std::string& testName)
{
    LOG_PRINT("  Test: %s (alpha=%.2f)\n", testName.c_str(), alphaValue);

    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;
    aclScalar* alpha = nullptr;
    void* workspaceAddr = nullptr;

    if (CreateAclTensor(selfData, selfShape, &selfDeviceAddr, dataType, &self) != 0) {
        TEST_FAIL();
        return;
    }
    if (CreateAclTensor(otherData, otherShape, &otherDeviceAddr, dataType, &other) != 0) {
        TEST_FAIL();
        goto cleanup_self;
    }
    if (CreateEmptyAclTensor<T>(selfShape, &outDeviceAddr, dataType, &out) != 0) {
        TEST_FAIL();
        goto cleanup_other;
    }
    alpha = CreateAclScalar(alphaValue, dataType);
    if (!alpha) {
        TEST_FAIL();
        goto cleanup_out;
    }

    {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        if (aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor) != ACL_SUCCESS) {
            TEST_FAIL();
            goto cleanup_alpha;
        }
        if (workspaceSize > 0) {
            if (aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                TEST_FAIL();
                goto cleanup_alpha;
            }
        }
        if (aclnnAdd(workspaceAddr, workspaceSize, executor, stream) != ACL_SUCCESS) {
            TEST_FAIL();
            goto cleanup_workspace;
        }
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            TEST_FAIL();
            goto cleanup_workspace;
        }

        int64_t outSize = GetShapeSize(selfShape);
        std::vector<T> resultData(outSize);
        if (aclrtMemcpy(
                resultData.data(), outSize * sizeof(T), outDeviceAddr, outSize * sizeof(T),
                ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            TEST_FAIL();
        } else {
            if (VerifySameShape(resultData, selfData, otherData, alphaValue)) {
                TEST_PASS();
            } else {
                TEST_FAIL();
            }
        }
    }

cleanup_workspace:
    if (workspaceAddr)
        aclrtFree(workspaceAddr);
cleanup_alpha:
    aclDestroyScalar(alpha);
cleanup_out:
    aclDestroyTensor(out);
    aclrtFree(outDeviceAddr);
cleanup_other:
    aclDestroyTensor(other);
    aclrtFree(otherDeviceAddr);
cleanup_self:
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
}

// ============== aclnnAdds 测试 ==============
template <typename T>
void RunAddsTest(
    const std::vector<T>& selfData, const std::vector<int64_t>& selfShape, T scalarValue, float alphaValue,
    aclDataType dataType, aclrtStream stream, const std::string& testName)
{
    LOG_PRINT("  Test: %s (scalar=%.2f, alpha=%.2f)\n", testName.c_str(), (float)scalarValue, alphaValue);

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclScalar* otherScalar = nullptr;
    aclScalar* alpha = nullptr;
    void* workspaceAddr = nullptr;

    if (CreateAclTensor(selfData, selfShape, &selfDeviceAddr, dataType, &self) != 0) {
        TEST_FAIL();
        return;
    }
    if (CreateEmptyAclTensor<T>(selfShape, &outDeviceAddr, dataType, &out) != 0) {
        TEST_FAIL();
        goto cleanup_self;
    }
    otherScalar = CreateAclScalar(scalarValue, dataType);
    alpha = CreateAclScalar(alphaValue, dataType);
    if (!otherScalar || !alpha) {
        TEST_FAIL();
        goto cleanup_out;
    }

    {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        if (aclnnAddsGetWorkspaceSize(self, otherScalar, alpha, out, &workspaceSize, &executor) != ACL_SUCCESS) {
            TEST_FAIL();
            goto cleanup_scalars;
        }
        if (workspaceSize > 0) {
            if (aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                TEST_FAIL();
                goto cleanup_scalars;
            }
        }
        if (aclnnAdds(workspaceAddr, workspaceSize, executor, stream) != ACL_SUCCESS) {
            TEST_FAIL();
            goto cleanup_workspace;
        }
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            TEST_FAIL();
            goto cleanup_workspace;
        }

        int64_t outSize = GetShapeSize(selfShape);
        std::vector<T> resultData(outSize);
        if (aclrtMemcpy(
                resultData.data(), outSize * sizeof(T), outDeviceAddr, outSize * sizeof(T),
                ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            TEST_FAIL();
        } else {
            bool pass = true;
            for (size_t i = 0; i < resultData.size(); i++) {
                T expected = selfData[i] + static_cast<T>(alphaValue * scalarValue);
                if (resultData[i] != expected) {
                    pass = false;
                    break;
                }
            }
            if (pass)
                TEST_PASS();
            else
                TEST_FAIL();
        }
    }

cleanup_workspace:
    if (workspaceAddr)
        aclrtFree(workspaceAddr);
cleanup_scalars:
    if (otherScalar)
        aclDestroyScalar(otherScalar);
    if (alpha)
        aclDestroyScalar(alpha);
cleanup_out:
    aclDestroyTensor(out);
    aclrtFree(outDeviceAddr);
cleanup_self:
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
}

// ============== aclnnAddV3 测试 ==============
template <typename T>
void RunAddV3Test(
    T selfScalar, const std::vector<T>& otherData, const std::vector<int64_t>& otherShape, float alphaValue,
    aclDataType dataType, aclrtStream stream, const std::string& testName)
{
    LOG_PRINT("  Test: %s (scalar=%.2f + alpha=%.2f * tensor)\n", testName.c_str(), (float)selfScalar, alphaValue);

    void* otherDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;
    aclScalar* selfScalarPtr = nullptr;
    aclScalar* alpha = nullptr;
    void* workspaceAddr = nullptr;

    if (CreateAclTensor(otherData, otherShape, &otherDeviceAddr, dataType, &other) != 0) {
        TEST_FAIL();
        return;
    }
    if (CreateEmptyAclTensor<T>(otherShape, &outDeviceAddr, dataType, &out) != 0) {
        TEST_FAIL();
        goto cleanup_other;
    }
    selfScalarPtr = CreateAclScalar(selfScalar, dataType);
    alpha = CreateAclScalar(alphaValue, dataType);
    if (!selfScalarPtr || !alpha) {
        TEST_FAIL();
        goto cleanup_out;
    }

    {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        if (aclnnAddV3GetWorkspaceSize(selfScalarPtr, other, alpha, out, &workspaceSize, &executor) != ACL_SUCCESS) {
            TEST_FAIL();
            goto cleanup_scalars;
        }
        if (workspaceSize > 0) {
            if (aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                TEST_FAIL();
                goto cleanup_scalars;
            }
        }
        if (aclnnAddV3(workspaceAddr, workspaceSize, executor, stream) != ACL_SUCCESS) {
            TEST_FAIL();
            goto cleanup_workspace;
        }
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            TEST_FAIL();
            goto cleanup_workspace;
        }

        int64_t outSize = GetShapeSize(otherShape);
        std::vector<T> resultData(outSize);
        if (aclrtMemcpy(
                resultData.data(), outSize * sizeof(T), outDeviceAddr, outSize * sizeof(T),
                ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            TEST_FAIL();
        } else {
            bool pass = true;
            for (size_t i = 0; i < resultData.size(); i++) {
                T expected = selfScalar + static_cast<T>(alphaValue * otherData[i]);
                if (resultData[i] != expected) {
                    pass = false;
                    break;
                }
            }
            if (pass)
                TEST_PASS();
            else
                TEST_FAIL();
        }
    }

cleanup_workspace:
    if (workspaceAddr)
        aclrtFree(workspaceAddr);
cleanup_scalars:
    if (selfScalarPtr)
        aclDestroyScalar(selfScalarPtr);
    if (alpha)
        aclDestroyScalar(alpha);
cleanup_out:
    aclDestroyTensor(out);
    aclrtFree(outDeviceAddr);
cleanup_other:
    aclDestroyTensor(other);
    aclrtFree(otherDeviceAddr);
}

// ============== aclnnInplaceAdd 测试 ==============
template <typename T>
void RunInplaceAddTest(
    std::vector<T>& selfData, const std::vector<int64_t>& selfShape, const std::vector<T>& otherData,
    const std::vector<int64_t>& otherShape, float alphaValue, aclDataType dataType, aclrtStream stream,
    const std::string& testName)
{
    LOG_PRINT("  Test: %s (alpha=%.2f)\n", testName.c_str(), alphaValue);

    void* selfDeviceAddr = nullptr;
    void* otherDeviceAddr = nullptr;
    aclTensor* selfRef = nullptr;
    aclTensor* other = nullptr;
    aclScalar* alpha = nullptr;
    void* workspaceAddr = nullptr;

    if (CreateAclTensor(selfData, selfShape, &selfDeviceAddr, dataType, &selfRef) != 0) {
        TEST_FAIL();
        return;
    }
    if (CreateAclTensor(otherData, otherShape, &otherDeviceAddr, dataType, &other) != 0) {
        TEST_FAIL();
        goto cleanup_self;
    }
    alpha = CreateAclScalar(alphaValue, dataType);
    if (!alpha) {
        TEST_FAIL();
        goto cleanup_other;
    }

    {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        if (aclnnInplaceAddGetWorkspaceSize(selfRef, other, alpha, &workspaceSize, &executor) != ACL_SUCCESS) {
            TEST_FAIL();
            goto cleanup_alpha;
        }
        if (workspaceSize > 0) {
            if (aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                TEST_FAIL();
                goto cleanup_alpha;
            }
        }
        if (aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, stream) != ACL_SUCCESS) {
            TEST_FAIL();
            goto cleanup_workspace;
        }
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            TEST_FAIL();
            goto cleanup_workspace;
        }

        int64_t outSize = GetShapeSize(selfShape);
        std::vector<T> resultData(outSize);
        if (aclrtMemcpy(
                resultData.data(), outSize * sizeof(T), selfDeviceAddr, outSize * sizeof(T),
                ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            TEST_FAIL();
        } else {
            bool pass = true;
            for (size_t i = 0; i < resultData.size(); i++) {
                T expected = selfData[i] + static_cast<T>(alphaValue * otherData[i % otherData.size()]);
                if (resultData[i] != expected) {
                    pass = false;
                    break;
                }
            }
            if (pass)
                TEST_PASS();
            else
                TEST_FAIL();
        }
    }

cleanup_workspace:
    if (workspaceAddr)
        aclrtFree(workspaceAddr);
cleanup_alpha:
    aclDestroyScalar(alpha);
cleanup_other:
    aclDestroyTensor(other);
    aclrtFree(otherDeviceAddr);
cleanup_self:
    aclDestroyTensor(selfRef);
    aclrtFree(selfDeviceAddr);
}

// ============== 主函数 ==============
int main()
{
    LOG_PRINT("\n========== Add Operator Ultimate Test Suite ==========\n\n");

    int32_t deviceId = 0;
    aclrtStream stream;
    if (Init(deviceId, &stream) != ACL_SUCCESS)
        return 1;

    // 1. 基础 float32 + 不同 alpha
    LOG_PRINT("--- 1. Basic float32 with different alpha ---\n");
    {
        std::vector<int64_t> shape = {4, 2};
        std::vector<float> selfData = {0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<float> otherData = {1, 1, 1, 2, 2, 2, 3, 3};
        RunAddTest<float>(selfData, shape, otherData, shape, 1.0f, ACL_FLOAT, stream, "float32_alpha1.0");
        RunAddTest<float>(selfData, shape, otherData, shape, 1.2f, ACL_FLOAT, stream, "float32_alpha1.2");
        RunAddTest<float>(selfData, shape, otherData, shape, 0.0f, ACL_FLOAT, stream, "float32_alpha0.0");
        RunAddTest<float>(selfData, shape, otherData, shape, -1.0f, ACL_FLOAT, stream, "float32_alpha_neg");
        RunAddTest<float>(selfData, shape, otherData, shape, 2.5f, ACL_FLOAT, stream, "float32_alpha2.5");
    }

    // 2. 不同数据类型
    LOG_PRINT("\n--- 2. Different dtypes ---\n");
    {
        std::vector<int64_t> shape = {2, 2};
        std::vector<int32_t> selfData = {1, 2, 3, 4};
        std::vector<int32_t> otherData = {5, 6, 7, 8};
        RunAddTest<int32_t>(selfData, shape, otherData, shape, 1.0f, ACL_INT32, stream, "int32");
    }
    {
        std::vector<int64_t> shape = {2, 2};
        std::vector<int64_t> selfData = {10, 20, 30, 40};
        std::vector<int64_t> otherData = {2, 3, 4, 5};
        RunAddTest<int64_t>(selfData, shape, otherData, shape, 1.0f, ACL_INT64, stream, "int64");
    }
    {
        std::vector<int64_t> shape = {2, 2};
        std::vector<int8_t> selfData = {1, 2, 3, 4};
        std::vector<int8_t> otherData = {2, 2, 2, 2};
        RunAddTest<int8_t>(selfData, shape, otherData, shape, 1.0f, ACL_INT8, stream, "int8");
    }
    {
        std::vector<int64_t> shape = {2, 2};
        std::vector<uint8_t> selfData = {10, 20, 30, 40};
        std::vector<uint8_t> otherData = {2, 2, 2, 2};
        RunAddTest<uint8_t>(selfData, shape, otherData, shape, 1.0f, ACL_UINT8, stream, "uint8");
    }

    // 3. 广播 Shape
    LOG_PRINT("\n--- 3. Broadcast shapes ---\n");
    {
        std::vector<int64_t> selfShape = {2, 3};
        std::vector<int64_t> otherShape = {3};
        std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
        std::vector<float> otherData = {10, 20, 30};
        RunAddTest<float>(selfData, selfShape, otherData, otherShape, 1.0f, ACL_FLOAT, stream, "broadcast_2x3_x_3");
    }
    {
        std::vector<int64_t> selfShape = {2, 1};
        std::vector<int64_t> otherShape = {1, 3};
        std::vector<float> selfData = {1, 2};
        std::vector<float> otherData = {10, 20, 30};
        RunAddTest<float>(selfData, selfShape, otherData, otherShape, 1.0f, ACL_FLOAT, stream, "broadcast_2x1_x_1x3");
    }

    // 4. aclnnAdds API
    LOG_PRINT("\n--- 4. aclnnAdds API ---\n");
    {
        std::vector<int64_t> shape = {3, 2};
        std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        RunAddsTest<float>(selfData, shape, 2.0f, 1.5f, ACL_FLOAT, stream, "adds_float32");
    }

    // 5. aclnnAddV3 API
    LOG_PRINT("\n--- 5. aclnnAddV3 API ---\n");
    {
        std::vector<int64_t> shape = {3, 2};
        std::vector<float> otherData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        RunAddV3Test<float>(10.0f, otherData, shape, 1.0f, ACL_FLOAT, stream, "addv3_float32");
        RunAddV3Test<float>(5.0f, otherData, shape, 2.0f, ACL_FLOAT, stream, "addv3_alpha2.0");
    }

    // 6. aclnnInplaceAdd API
    LOG_PRINT("\n--- 6. aclnnInplaceAdd API ---\n");
    {
        std::vector<int64_t> shape = {3, 2};
        std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> otherData = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
        RunInplaceAddTest<float>(selfData, shape, otherData, shape, 1.0f, ACL_FLOAT, stream, "inplace_add_float32");
    }

    // 7. 边界值
    LOG_PRINT("\n--- 7. Boundary values ---\n");
    {
        std::vector<int64_t> shape = {3};
        std::vector<float> selfData = {0.0f, -1.0f, 100.0f};
        std::vector<float> otherData = {5.0f, -2.0f, 0.0f};
        RunAddTest<float>(selfData, shape, otherData, shape, 1.0f, ACL_FLOAT, stream, "zero_negative");
    }

    LOG_PRINT("\n========== Test Summary ==========\n");
    LOG_PRINT("Total: %d, Passed: %d, Failed: %d\n", g_total_tests, g_passed_tests, g_failed_tests);
    LOG_PRINT("===================================\n\n");

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return (g_failed_tests == 0) ? 0 : 1;
}