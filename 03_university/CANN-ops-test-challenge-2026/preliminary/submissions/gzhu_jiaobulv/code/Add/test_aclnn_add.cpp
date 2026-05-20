/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <cstdint>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"

#define CHECK_RET(cond, return_expr) \
    do                               \
    {                                \
        if (!(cond))                 \
        {                            \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do                                  \
    {                                   \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// ======================== 辅助函数 ========================

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (auto d : shape)
        size *= d;
    return size;
}

// 浮点数比较（支持 NaN/Inf 和容差）
bool AlmostEqual(double expected, double actual, double atol = 1e-5, double rtol = 1e-5)
{
    if (std::isnan(expected) && std::isnan(actual))
        return true;
    if (std::isinf(expected) && std::isinf(actual))
        return (expected > 0) == (actual > 0);
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

// ACL 初始化
int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// 创建 ACL tensor（支持空 tensor）
template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape,
                    void **deviceAddr,
                    aclDataType dataType,
                    aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 计算连续 strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = (int64_t)shape.size() - 2; i >= 0; --i)
    {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    if (size == 0)
    {
        *deviceAddr = nullptr;
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                  aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
        return (*tensor == nullptr) ? 1 : 0;
    }
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// 创建非连续 tensor（用于测试）
template <typename T>
int CreateNonContiguousAclTensor(const std::vector<T> &hostData,
                                 const std::vector<int64_t> &shape,
                                 const std::vector<int64_t> &strides,
                                 void **deviceAddr,
                                 aclDataType dataType,
                                 aclTensor **tensor)
{
    auto size = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return ret;
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS)
        return ret;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// ======================== 通用测试模板 ========================

// 通用加法测试（同类型，alpha 可配置）
template <typename T>
int RunGenericAddTest(const std::string &testName,
                      aclrtStream stream,
                      aclDataType dataType,
                      const std::vector<int64_t> &shape1,
                      const std::vector<int64_t> &shape2,
                      const std::vector<int64_t> &outShape,
                      const std::vector<T> &host1,
                      const std::vector<T> &host2,
                      T alphaValue,
                      const std::vector<T> &expectedHost,
                      double atol = 1e-5,
                      double rtol = 1e-5)
{
    void *dev1 = nullptr, *dev2 = nullptr, *outDev = nullptr, *workspace = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *out = nullptr;
    aclScalar *alpha = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;
    int failed = 0;
    int retCode = 0;

    std::vector<T> outHost(expectedHost.size(), 0);

    if (CreateAclTensor(host1, shape1, &dev1, dataType, &t1) != 0)
    {
        retCode = 1;
        goto cleanup;
    }
    if (CreateAclTensor(host2, shape2, &dev2, dataType, &t2) != 0)
    {
        retCode = 1;
        goto cleanup;
    }
    if (CreateAclTensor(outHost, outShape, &outDev, dataType, &out) != 0)
    {
        retCode = 1;
        goto cleanup;
    }

    alpha = aclCreateScalar(&alphaValue, dataType);
    if (alpha == nullptr)
    {
        retCode = 1;
        goto cleanup;
    }

    if (aclnnAddGetWorkspaceSize(t1, t2, alpha, out, &workspaceSize, &executor) != ACL_SUCCESS)
    {
        retCode = 1;
        goto cleanup;
    }
    if (workspaceSize > 0)
    {
        if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
        {
            retCode = 1;
            goto cleanup;
        }
    }
    if (aclnnAdd(workspace, workspaceSize, executor, stream) != ACL_SUCCESS)
    {
        retCode = 1;
        goto cleanup;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
    {
        retCode = 1;
        goto cleanup;
    }

    if (aclrtMemcpy(outHost.data(), outHost.size() * sizeof(T), outDev,
                    outHost.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
    {
        retCode = 1;
        goto cleanup;
    }

    for (size_t i = 0; i < expectedHost.size(); ++i)
    {
        double expVal = (double)expectedHost[i];
        double actVal = (double)outHost[i];
        if (!AlmostEqual(expVal, actVal, atol, rtol))
        {
            LOG_PRINT("[FAIL] %s idx %zu: expected %f, got %f\n", testName.c_str(), i, expVal, actVal);
            failed++;
        }
    }

cleanup:
    if (retCode != 0)
    {
        LOG_PRINT("[ERROR] %s terminated early due to ACL API error.\n", testName.c_str());
    }
    else if (failed == 0)
    {
        LOG_PRINT("[PASS] %s\n", testName.c_str());
    }
    if (workspace)
        aclrtFree(workspace);
    if (t1)
        aclDestroyTensor(t1);
    if (t2)
        aclDestroyTensor(t2);
    if (out)
        aclDestroyTensor(out);
    if (alpha)
        aclDestroyScalar(alpha);
    if (dev1)
        aclrtFree(dev1);
    if (dev2)
        aclrtFree(dev2);
    if (outDev)
        aclrtFree(outDev);
    return failed + retCode;
}

// 原地加法测试（self 和 out 是同一个 tensor）
template <typename T>
int RunGenericInplaceAddTest(const std::string &testName,
                             aclrtStream stream,
                             aclDataType dataType,
                             const std::vector<int64_t> &shape,
                             const std::vector<T> &selfHost,
                             const std::vector<T> &otherHost,
                             T alphaValue,
                             const std::vector<T> &expectedHost,
                             double atol = 1e-5,
                             double rtol = 1e-5)
{
    void *selfDev = nullptr, *otherDev = nullptr, *workspace = nullptr;
    aclTensor *self = nullptr, *other = nullptr;
    aclScalar *alpha = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;
    int failed = 0;
    int retCode = 0;

    if (CreateAclTensor(selfHost, shape, &selfDev, dataType, &self) != 0)
    {
        retCode = 1;
        goto cleanup;
    }
    if (CreateAclTensor(otherHost, shape, &otherDev, dataType, &other) != 0)
    {
        retCode = 1;
        goto cleanup;
    }

    alpha = aclCreateScalar(&alphaValue, dataType);
    if (alpha == nullptr)
    {
        retCode = 1;
        goto cleanup;
    }

    // 原地加法：将 self 作为 out 传入
    if (aclnnAddGetWorkspaceSize(self, other, alpha, self, &workspaceSize, &executor) != ACL_SUCCESS)
    {
        retCode = 1;
        goto cleanup;
    }
    if (workspaceSize > 0)
    {
        if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
        {
            retCode = 1;
            goto cleanup;
        }
    }
    if (aclnnAdd(workspace, workspaceSize, executor, stream) != ACL_SUCCESS)
    {
        retCode = 1;
        goto cleanup;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
    {
        retCode = 1;
        goto cleanup;
    }

    std::vector<T> outHost(expectedHost.size(), 0);
    if (aclrtMemcpy(outHost.data(), outHost.size() * sizeof(T), selfDev,
                    outHost.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
    {
        retCode = 1;
        goto cleanup;
    }

    for (size_t i = 0; i < expectedHost.size(); ++i)
    {
        double expVal = (double)expectedHost[i];
        double actVal = (double)outHost[i];
        if (!AlmostEqual(expVal, actVal, atol, rtol))
        {
            LOG_PRINT("[FAIL] %s idx %zu: expected %f, got %f\n", testName.c_str(), i, expVal, actVal);
            failed++;
        }
    }

cleanup:
    if (retCode != 0)
    {
        LOG_PRINT("[ERROR] %s terminated early.\n", testName.c_str());
    }
    else if (failed == 0)
    {
        LOG_PRINT("[PASS] %s\n", testName.c_str());
    }
    if (workspace)
        aclrtFree(workspace);
    if (self)
        aclDestroyTensor(self);
    if (other)
        aclDestroyTensor(other);
    if (alpha)
        aclDestroyScalar(alpha);
    if (selfDev)
        aclrtFree(selfDev);
    if (otherDev)
        aclrtFree(otherDev);
    return failed + retCode;
}

// ======================== 专用测试函数 ========================

// float16 转换函数
float Float16ToFloat(uint16_t fp16)
{
    uint16_t sign = (fp16 >> 15) & 0x1;
    uint16_t exponent = (fp16 >> 10) & 0x1F;
    uint16_t mantissa = fp16 & 0x3FF;
    if (exponent == 0)
    {
        if (mantissa == 0)
            return sign ? -0.0f : 0.0f;
        float value = (float)mantissa / 1024.0f;
        value = ldexp(value, -14);
        return sign ? -value : value;
    }
    else if (exponent == 0x1F)
    {
        if (mantissa == 0)
            return sign ? -INFINITY : INFINITY;
        return NAN;
    }
    else
    {
        float value = ldexpf((float)(mantissa | 0x400), exponent - 25);
        return sign ? -value : value;
    }
}

// float16 加法测试（专用，因为需要转换）
int RunFloat16AddTest(aclrtStream stream)
{
    int totalFailed = 0;
    std::vector<int64_t> shape = {2};
    std::vector<uint16_t> selfHost = {0x3C00, 0x4000};  // 1.0, 2.0
    std::vector<uint16_t> otherHost = {0x4200, 0x4200}; // 3.0, 3.0
    std::vector<uint16_t> outHost(2, 0);
    std::vector<float> expected = {4.0f, 5.0f}; // 1+3=4, 2+3=5
    float alphaVal = 1.0f;

    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr, *workspace = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    aclScalar *alpha = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;
    int failed = 0;

    if (CreateAclTensor(selfHost, shape, &selfDev, ACL_FLOAT16, &self) != 0)
        goto cleanup;
    if (CreateAclTensor(otherHost, shape, &otherDev, ACL_FLOAT16, &other) != 0)
        goto cleanup;
    if (CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT16, &out) != 0)
        goto cleanup;
    alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
    if (alpha == nullptr)
        goto cleanup;

    if (aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor) != ACL_SUCCESS)
        goto cleanup;
    if (workspaceSize > 0 && aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
        goto cleanup;
    if (aclnnAdd(workspace, workspaceSize, executor, stream) != ACL_SUCCESS)
        goto cleanup;
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost.data(), outHost.size() * sizeof(uint16_t), outDev,
                outHost.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);

    for (size_t i = 0; i < expected.size(); ++i)
    {
        float actual = Float16ToFloat(outHost[i]);
        if (!AlmostEqual(expected[i], actual, 1e-3, 1e-3))
        {
            LOG_PRINT("[FAIL] float16_add idx %zu: expected %f, got %f\n", i, expected[i], actual);
            failed++;
        }
    }
    if (failed == 0)
        LOG_PRINT("[PASS] float16_add\n");
    totalFailed += failed;

cleanup:
    if (workspace)
        aclrtFree(workspace);
    if (self)
        aclDestroyTensor(self);
    if (other)
        aclDestroyTensor(other);
    if (out)
        aclDestroyTensor(out);
    if (alpha)
        aclDestroyScalar(alpha);
    if (selfDev)
        aclrtFree(selfDev);
    if (otherDev)
        aclrtFree(otherDev);
    if (outDev)
        aclrtFree(outDev);
    return totalFailed;
}

// 异常测试（不支持 dtype、广播不匹配、nullptr 等）
int RunExceptionTests(aclrtStream stream)
{
    int failed = 0;
    LOG_PRINT("\n--- 异常测试 ---\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> fData = {1, 2, 3, 4};
    void *devF = nullptr, *devU32 = nullptr;
    aclTensor *tF = nullptr, *tU32 = nullptr;
    aclScalar *alpha = aclCreateScalar(&fData[0], ACL_FLOAT);
    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;

    // 1. 不支持的数据类型 (UINT32)
    std::vector<uint32_t> u32Data = {1, 2, 3, 4};
    if (CreateAclTensor(u32Data, shape, &devU32, ACL_UINT32, &tU32) == 0 && tU32 != nullptr)
    {
        if (aclnnAddGetWorkspaceSize(tU32, tU32, alpha, tU32, &wsSize, &exec) == ACL_SUCCESS)
        {
            LOG_PRINT("[FAIL] unsupported dtype UINT32 was accepted\n");
            failed++;
        }
        else
        {
            LOG_PRINT("[PASS] unsupported dtype UINT32 rejected\n");
        }
        aclDestroyTensor(tU32);
        aclrtFree(devU32);
    }
    else
    {
        LOG_PRINT("[PASS] unsupported dtype UINT32 rejected at creation\n");
    }

    // 2. 无法广播的 shape 组合
    if (CreateAclTensor(fData, shape, &devF, ACL_FLOAT, &tF) == 0)
    {
        std::vector<int64_t> badShape = {4, 5};
        void *devBad = nullptr;
        aclTensor *tBad = nullptr;
        std::vector<float> badData(20, 1.0f);
        if (CreateAclTensor(badData, badShape, &devBad, ACL_FLOAT, &tBad) == 0 && tBad != nullptr)
        {
            if (aclnnAddGetWorkspaceSize(tF, tBad, alpha, tF, &wsSize, &exec) == ACL_SUCCESS)
            {
                LOG_PRINT("[FAIL] non-broadcastable shapes were accepted\n");
                failed++;
            }
            else
            {
                LOG_PRINT("[PASS] non-broadcastable shapes rejected\n");
            }
            aclDestroyTensor(tBad);
            aclrtFree(devBad);
        }
        aclDestroyTensor(tF);
        aclrtFree(devF);
    }

    // 3. nullptr 参数
    auto ret = aclnnAddGetWorkspaceSize(nullptr, nullptr, alpha, nullptr, &wsSize, &exec);
    if (ret == ACL_SUCCESS)
    {
        LOG_PRINT("[FAIL] nullptr inputs were accepted\n");
        failed++;
    }
    else
    {
        LOG_PRINT("[PASS] nullptr inputs rejected\n");
    }

    if (alpha)
        aclDestroyScalar(alpha);
    return failed;
}

// ======================== 主函数 ========================
int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed\n"); return ret);

    int totalFailed = 0;
    LOG_PRINT("========== Add 算子覆盖率测试 ==========\n");

    // ========== 第一组：普通加法（alpha=1）==========
    LOG_PRINT("\n--- 1. 普通加法 (alpha=1) ---\n");
    totalFailed += RunGenericAddTest<float>(
        "add_float32_basic", stream, ACL_FLOAT,
        {4, 2}, {4, 2}, {4, 2},
        {0, 1, 2, 3, 4, 5, 6, 7}, {1, 1, 1, 2, 2, 2, 3, 3}, 1.0f,
        {1, 2, 3, 5, 6, 7, 9, 10});
    totalFailed += RunGenericAddTest<int32_t>(
        "add_int32_basic", stream, ACL_INT32,
        {2, 2}, {2, 2}, {2, 2},
        {1, 2, 3, 4}, {2, 2, 2, 2}, 1,
        {3, 4, 5, 6}, 0.0, 0.0);
    totalFailed += RunGenericAddTest<uint8_t>(
        "add_uint8_basic", stream, ACL_UINT8,
        {2, 2}, {2, 2}, {2, 2},
        {1, 2, 3, 10}, {2, 3, 4, 5}, 1,
        {3, 5, 7, 15}, 0.0, 0.0);

    // ========== 第二组：带 alpha 的加法 ==========
    LOG_PRINT("\n--- 2. 带 alpha 的加法 ---\n");
    totalFailed += RunGenericAddTest<float>(
        "add_float32_alpha2", stream, ACL_FLOAT,
        {2, 2}, {2, 2}, {2, 2},
        {1, 2, 3, 4}, {1, 1, 1, 1}, 2.0f,
        {3, 4, 5, 6});
    totalFailed += RunGenericAddTest<int32_t>(
        "add_int32_alpha_negative", stream, ACL_INT32,
        {2, 2}, {2, 2}, {2, 2},
        {10, 20, 30, 40}, {1, 2, 3, 4}, -2,
        {8, 16, 24, 32}, 0.0, 0.0);

    // ========== 第三组：广播测试 ==========
    LOG_PRINT("\n--- 3. 广播测试 ---\n");
    totalFailed += RunGenericAddTest<float>(
        "add_broadcast_2x3_3", stream, ACL_FLOAT,
        {2, 3}, {3}, {2, 3},
        {1, 2, 3, 4, 5, 6}, {10, 20, 30}, 1.0f,
        {11, 22, 33, 14, 25, 36});
    totalFailed += RunGenericAddTest<float>(
        "add_broadcast_3d", stream, ACL_FLOAT,
        {2, 1, 4}, {1, 3, 4}, {2, 3, 4},
        {1, 1, 1, 1, 2, 2, 2, 2}, {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}, 1.0f,
        {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6}, 1e-5, 1e-5);

    // ========== 第四组：原地加法 (self == out) ==========
    LOG_PRINT("\n--- 4. 原地加法测试 ---\n");
    totalFailed += RunGenericInplaceAddTest<float>(
        "inplace_add_float32", stream, ACL_FLOAT,
        {2, 2}, {1, 2, 3, 4}, {2, 2, 2, 2}, 1.0f,
        {3, 4, 5, 6});
    totalFailed += RunGenericInplaceAddTest<int32_t>(
        "inplace_add_int32", stream, ACL_INT32,
        {4}, {10, 20, 30, 40}, {1, 2, 3, 4}, 1,
        {11, 22, 33, 44}, 0.0, 0.0);

    // ========== 第五组：边界值与特殊值 ==========
    LOG_PRINT("\n--- 5. 边界值与特殊值 ---\n");
    totalFailed += RunGenericAddTest<float>(
        "add_nan_inf", stream, ACL_FLOAT,
        {4}, {4}, {4},
        {INFINITY, -INFINITY, NAN, 5.0f},
        {2.0f, 3.0f, 2.0f, NAN}, 1.0f,
        {INFINITY, -INFINITY, NAN, NAN},
        1e-5, 1e-5);
    totalFailed += RunGenericAddTest<int32_t>(
        "add_int32_limits", stream, ACL_INT32,
        {3}, {3}, {3},
        // 输入分别设为: INT_MAX - 1, INT_MIN + 1, 100
        {std::numeric_limits<int32_t>::max() - 1, std::numeric_limits<int32_t>::min() + 1, 100},
        // 对应加上: 1, -1, 0
        {1, -1, 0}, 1,
        // 结果恰好达到绝对极限：INT_MAX, INT_MIN, 100
        {std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min(), 100},
        0.0, 0.0);

    // ========== 第六组：float16 专用测试 ==========
    LOG_PRINT("\n--- 6. float16 测试 ---\n");
    totalFailed += RunFloat16AddTest(stream);

    // ========== 第七组：非连续内存测试 ==========
    LOG_PRINT("\n--- 7. 非连续内存测试 ---\n");
    {
        std::vector<int64_t> shape = {2, 2};
        std::vector<int64_t> badStrides = {4, 1};
        std::vector<float> data1 = {1, 2, 0, 0, 3, 4, 0, 0};
        std::vector<float> data2 = {2, 2, 0, 0, 2, 2, 0, 0};
        std::vector<float> outData(8, 0);
        std::vector<float> expected = {3, 4, 0, 0, 5, 6, 0, 0};
        void *dev1 = nullptr, *dev2 = nullptr, *outDev = nullptr;
        aclTensor *t1 = nullptr, *t2 = nullptr, *out = nullptr;
        if (CreateNonContiguousAclTensor(data1, shape, badStrides, &dev1, ACL_FLOAT, &t1) == 0 &&
            CreateNonContiguousAclTensor(data2, shape, badStrides, &dev2, ACL_FLOAT, &t2) == 0 &&
            CreateNonContiguousAclTensor(outData, shape, badStrides, &outDev, ACL_FLOAT, &out) == 0)
        {
            uint64_t ws = 0;
            aclOpExecutor *exec = nullptr;
            void *workspace = nullptr;
            aclScalar *alpha = aclCreateScalar(&data1[0], ACL_FLOAT); // dummy alpha=1
            if (aclnnAddGetWorkspaceSize(t1, t2, alpha, out, &ws, &exec) == ACL_SUCCESS)
            {
                if (ws > 0)
                    aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
                if (aclnnAdd(workspace, ws, exec, stream) == ACL_SUCCESS)
                {
                    aclrtSynchronizeStream(stream);
                    aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
                    int failed = 0;
                    for (size_t i = 0; i < expected.size(); ++i)
                    {
                        if (!AlmostEqual(expected[i], outData[i]))
                        {
                            failed++;
                            LOG_PRINT("[FAIL] non_contiguous idx %zu\n", i);
                        }
                    }
                    if (failed == 0)
                        LOG_PRINT("[PASS] non_contiguous_add\n");
                    else
                        totalFailed += failed;
                }
                if (workspace)
                    aclrtFree(workspace);
            }
            if (alpha)
                aclDestroyScalar(alpha);
            aclDestroyTensor(t1);
            aclDestroyTensor(t2);
            aclDestroyTensor(out);
            aclrtFree(dev1);
            aclrtFree(dev2);
            aclrtFree(outDev);
        }
    }

    // ========== 第八组：异常测试 ==========
    totalFailed += RunExceptionTests(stream);

    // ========== 汇总 ==========
    LOG_PRINT("\n====================================================\n");
    if (totalFailed == 0)
    {
        LOG_PRINT("ALL TESTS PASSED\n");
    }
    else
    {
        LOG_PRINT("TOTAL FAILED = %d\n", totalFailed);
    }
    LOG_PRINT("====================================================\n");

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return totalFailed;
}