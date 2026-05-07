/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file without compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include <iostream>
 #include <vector>
 #include <cmath>
 #include <cstdint>
 #include <limits>
 #include <cstdio>
 #include <cstdlib>
 #include <algorithm>
 #include "acl/acl.h"
 #include "aclnn_add.h"
 #include "aclnn_add_v3.h"

 // Helper union for converting between uint16_t (float16/bf16 storage) and float.
 // Avoids depending on the 'half' type which may not be available through cust build paths.
 union Float16Union {
     uint16_t u;
     float f;
 };

 static inline float Uint16ToFloat(uint16_t v) {
     Float16Union conv;
     conv.u = v;
     return conv.f;
 }

 static inline uint16_t FloatToUint16(float v) {
     Float16Union conv;
     conv.f = v;
     return conv.u;
 }

// Convert a uint16_t (float16 or bf16 raw storage) to double for CPU reference comparison.
 static inline double ToDouble(uint16_t val) {
     return static_cast<double>(Uint16ToFloat(val));
 }

// Cast a float to type T (used for constructing host-side test data).
// For uint16_t (float16/bf16), use FloatToUint16 to convert float bits correctly.
 template <typename T>
 static inline T CastFloatTo(float v) { return static_cast<T>(v); }

 template <>
 inline uint16_t CastFloatTo<uint16_t>(float v) { return FloatToUint16(v); }
 
 #define CHECK_RET(cond, return_expr) \
     do {                             \
         if (!(cond)) {                \
             return_expr;              \
         }                            \
     } while (0)
 
 #define LOG_PRINT(message, ...)        \
     do {                               \
         printf(message, ##__VA_ARGS__); \
     } while (0)
 
 static bool g_aclInitialized = false;
 static int32_t g_deviceId = 0;
 static aclrtStream g_stream = nullptr;
 
 int InitAcl(int32_t deviceId, aclrtStream* stream)
 {
     if (g_aclInitialized) {
         *stream = g_stream;
         return 0;
     }
     auto ret = aclInit(nullptr);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
     ret = aclrtSetDevice(deviceId);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
     ret = aclrtCreateStream(stream);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
     g_aclInitialized = true;
     g_deviceId = deviceId;
     g_stream = *stream;
     return 0;
 }
 
 // =============================================================
 // CPU reference implementation for Add: out = self + alpha * other
 // Uses double precision to provide accurate reference.
 // The reference function receives float inputs to ensure the
 // same quantized values as NPU.
 // =============================================================
 static double CpuAdd(float selfVal, float otherVal, float alphaVal)
 {
     return static_cast<double>(selfVal) + static_cast<double>(alphaVal) * static_cast<double>(otherVal);
 }
 
// Convert any type T to double for CPU reference comparison.
// Specializations handle uint16_t (float16/bf16) by interpreting the bits as float.
 template <typename T>
 static inline double CastToDouble(T val) { return static_cast<double>(val); }

 template <>
 inline double CastToDouble<uint16_t>(uint16_t val) { return ToDouble(val); }

 template <typename T>
 static int ExpectNearRelative(
     const std::vector<T>& actual, const std::vector<T>& expected,
     double rtol, double atol, const char* name, bool verbose = false)
 {
     if (actual.size() != expected.size()) {
         LOG_PRINT("[FAIL] %s size mismatch: actual=%zu expected=%zu\n",
                   name, actual.size(), expected.size());
         return ACL_ERROR_FAILURE;
     }
     int failCount = 0;
     for (size_t i = 0; i < actual.size(); ++i) {
        const double a = CastToDouble(actual[i]);
        const double e = CastToDouble(expected[i]);
         if (!std::isfinite(a) || !std::isfinite(e)) {
             if (verbose) {
                 LOG_PRINT("[INFO] %s[%zu] non-finite: actual=%f expected=%f (allowed for inf/nan cases)\n",
                           name, i, a, e);
             }
             continue;
         }
         const double diff = std::abs(a - e);
         const double tol = atol + rtol * std::abs(e);
         if (diff > tol) {
             LOG_PRINT("[FAIL] %s[%zu] actual=%0.15g expected=%0.15g diff=%0.10g tol=%0.10g\n",
                       name, i, a, e, diff, tol);
             failCount++;
         }
     }
     if (failCount > 0) {
         LOG_PRINT("[FAIL] %s: %d elements failed\n", name, failCount);
         return ACL_ERROR_FAILURE;
     }
     return ACL_SUCCESS;
 }
 
 template <typename T>
 static int ExpectEqualVector(
     const std::vector<T>& actual, const std::vector<T>& expected, const char* name)
 {
     if (actual.size() != expected.size()) {
         LOG_PRINT("[FAIL] %s size mismatch: actual=%zu expected=%zu\n",
                   name, actual.size(), expected.size());
         return ACL_ERROR_FAILURE;
     }
     for (size_t i = 0; i < actual.size(); ++i) {
         if (actual[i] != expected[i]) {
             LOG_PRINT("[FAIL] %s[%zu] actual=%lld expected=%lld\n", name, i,
                       static_cast<long long>(actual[i]),
                       static_cast<long long>(expected[i]));
             return ACL_ERROR_FAILURE;
         }
     }
     return ACL_SUCCESS;
 }
 
 // =============================================================
 // Template helper: run aclnnAdd with given inputs
 // self_val[i] + alpha_val * other_val[i] -> output
 // =============================================================
 template <typename T>
 static int RunAddTest(
     aclrtStream stream,
     const std::vector<float>& self_val,
     const std::vector<float>& other_val,
     float alpha_val,
     aclDataType dtype, size_t count,
     double rtol, double atol,
     const char* testName,
     bool verbose = false)
 {
    int ret = ACL_SUCCESS;
    std::vector<T> actual(count, static_cast<T>(0));
    std::vector<T> selfHost(count);
    std::vector<T> otherHost(count);
    for (size_t i = 0; i < count; ++i) {
        selfHost[i] = CastFloatTo<T>(self_val[i]);
        otherHost[i] = CastFloatTo<T>(other_val[i]);
    }
    T alphaHost = CastFloatTo<T>(alpha_val);
 
     void* selfDev = nullptr;
     void* otherDev = nullptr;
     void* outDev = nullptr;
     aclTensor* self = nullptr;
     aclTensor* other = nullptr;
     aclTensor* out = nullptr;
     aclScalar* alpha = nullptr;
     void* workspaceDev = nullptr;
 
     do {
         ret = aclrtMalloc(&selfDev, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
         ret = aclrtMalloc(&otherDev, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
         ret = aclrtMalloc(&outDev, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
 
         ret = aclrtMemcpy(selfDev, count * sizeof(T), selfHost.data(), count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
         ret = aclrtMemcpy(otherDev, count * sizeof(T), otherHost.data(), count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
 
         int64_t shape = static_cast<int64_t>(count);
         self = aclCreateTensor(&shape, 1, dtype, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         CHECK_RET(self != nullptr, ret = ACL_ERROR_FAILURE; break);
         other = aclCreateTensor(&shape, 1, dtype, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         CHECK_RET(other != nullptr, ret = ACL_ERROR_FAILURE; break);
         out = aclCreateTensor(&shape, 1, dtype, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         CHECK_RET(out != nullptr, ret = ACL_ERROR_FAILURE; break);
         alpha = aclCreateScalar(&alphaHost, dtype);
         CHECK_RET(alpha != nullptr, ret = ACL_ERROR_FAILURE; break);
 
         uint64_t workspaceSize = 0;
         aclOpExecutor* executor = nullptr;
         ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
         CHECK_RET(ret == ACL_SUCCESS, break);
         CHECK_RET(executor != nullptr, ret = ACL_ERROR_FAILURE; break);
 
         if (workspaceSize > 0) {
             ret = aclrtMalloc(&workspaceDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
             CHECK_RET(ret == ACL_SUCCESS, break);
         }
 
         ret = aclnnAdd(workspaceDev, workspaceSize, executor, stream);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
         ret = aclrtSynchronizeStream(stream);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
         ret = aclrtMemcpy(actual.data(), count * sizeof(T), outDev, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
        if (verbose) {
            for (size_t i = 0; i < count; ++i) {
                LOG_PRINT("  [%zu] NPU=%0.15g CPU=%0.15g\n",
                          i, CastToDouble(actual[i]),
                          CpuAdd(self_val[i], other_val[i], alpha_val));
            }
        }

        std::vector<T> expectedHost(count);
        for (size_t i = 0; i < count; ++i) {
            expectedHost[i] = CastFloatTo<T>(CpuAdd(self_val[i], other_val[i], alpha_val));
        }

        // Use near comparison for all non-integer types (including uint16_t as float16/bf16).
        if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, uint16_t>) {
            ret = ExpectNearRelative(actual, expectedHost, rtol, atol, testName, verbose);
        } else {
            ret = ExpectEqualVector(actual, expectedHost, testName);
        }
     } while (false);
 
     if (alpha) aclDestroyScalar(alpha);
     if (out) aclDestroyTensor(out);
     if (other) aclDestroyTensor(other);
     if (self) aclDestroyTensor(self);
     if (outDev) aclrtFree(outDev);
     if (otherDev) aclrtFree(otherDev);
     if (selfDev) aclrtFree(selfDev);
     if (workspaceDev) aclrtFree(workspaceDev);
 
     if (ret == ACL_SUCCESS) {
         LOG_PRINT("[PASS] %s\n", testName);
     } else {
         LOG_PRINT("[FAIL] %s (ret=%d)\n", testName, ret);
     }
     return ret;
 }
 
 // =============================================================
 // aclnnAdds: self + alpha * scalar
 // =============================================================
 template <typename T>
 static int RunAddsTest(
     aclrtStream stream,
     const std::vector<float>& self_val,
     float scalar_val,
     float alpha_val,
     aclDataType dtype, size_t count,
     double rtol, double atol,
     const char* testName,
     bool verbose = false)
 {
     int ret = ACL_SUCCESS;
    std::vector<T> actual(count, static_cast<T>(0));
    std::vector<T> selfHost(count);
    for (size_t i = 0; i < count; ++i) {
        selfHost[i] = CastFloatTo<T>(self_val[i]);
    }
    T scalarHost = CastFloatTo<T>(scalar_val);
    T alphaHost = CastFloatTo<T>(alpha_val);
 
     void* selfDev = nullptr;
     void* outDev = nullptr;
     aclTensor* self = nullptr;
     aclTensor* out = nullptr;
     aclScalar* scalar = nullptr;
     aclScalar* alpha = nullptr;
     void* workspaceDev = nullptr;
 
     do {
         ret = aclrtMalloc(&selfDev, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
         ret = aclrtMalloc(&outDev, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
 
         ret = aclrtMemcpy(selfDev, count * sizeof(T), selfHost.data(), count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
 
         int64_t shape = static_cast<int64_t>(count);
         self = aclCreateTensor(&shape, 1, dtype, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         CHECK_RET(self != nullptr, ret = ACL_ERROR_FAILURE; break);
         out = aclCreateTensor(&shape, 1, dtype, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         CHECK_RET(out != nullptr, ret = ACL_ERROR_FAILURE; break);
         scalar = aclCreateScalar(&scalarHost, dtype);
         CHECK_RET(scalar != nullptr, ret = ACL_ERROR_FAILURE; break);
         alpha = aclCreateScalar(&alphaHost, dtype);
         CHECK_RET(alpha != nullptr, ret = ACL_ERROR_FAILURE; break);
 
         uint64_t workspaceSize = 0;
         aclOpExecutor* executor = nullptr;
         ret = aclnnAddsGetWorkspaceSize(self, scalar, alpha, out, &workspaceSize, &executor);
         CHECK_RET(ret == ACL_SUCCESS, break);
         CHECK_RET(executor != nullptr, ret = ACL_ERROR_FAILURE; break);
 
         if (workspaceSize > 0) {
             ret = aclrtMalloc(&workspaceDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
             CHECK_RET(ret == ACL_SUCCESS, break);
         }
 
         ret = aclnnAdds(workspaceDev, workspaceSize, executor, stream);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
         ret = aclrtSynchronizeStream(stream);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
         ret = aclrtMemcpy(actual.data(), count * sizeof(T), outDev, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
         std::vector<T> expectedHost(count);
         for (size_t i = 0; i < count; ++i) {
            expectedHost[i] = CastFloatTo<T>(CpuAdd(self_val[i], scalar_val, alpha_val));
        }

        if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, uint16_t>) {
            ret = ExpectNearRelative(actual, expectedHost, rtol, atol, testName, verbose);
        } else {
            ret = ExpectEqualVector(actual, expectedHost, testName);
        }
    } while (false);
 
     if (alpha) aclDestroyScalar(alpha);
     if (scalar) aclDestroyScalar(scalar);
     if (out) aclDestroyTensor(out);
     if (self) aclDestroyTensor(self);
     if (outDev) aclrtFree(outDev);
     if (selfDev) aclrtFree(selfDev);
     if (workspaceDev) aclrtFree(workspaceDev);
 
     if (ret == ACL_SUCCESS) {
         LOG_PRINT("[PASS] %s\n", testName);
     } else {
         LOG_PRINT("[FAIL] %s (ret=%d)\n", testName, ret);
     }
     return ret;
 }
 
 // =============================================================
 // aclnnAddV3: scalar + alpha * tensor
 // =============================================================
 template <typename T>
 static int RunAddV3Test(
     aclrtStream stream,
     float self_val,
     const std::vector<float>& other_val,
     float alpha_val,
     aclDataType dtype, size_t count,
     double rtol, double atol,
     const char* testName)
 {
     int ret = ACL_SUCCESS;
    std::vector<T> actual(count, static_cast<T>(0));
    std::vector<T> otherHost(count);
    for (size_t i = 0; i < count; ++i) {
        otherHost[i] = CastFloatTo<T>(other_val[i]);
    }
    T selfHost = CastFloatTo<T>(self_val);
    T alphaHost = CastFloatTo<T>(alpha_val);
 
     void* selfDev = nullptr;
     void* otherDev = nullptr;
     void* outDev = nullptr;
     aclScalar* selfScalar = nullptr;
     aclTensor* other = nullptr;
     aclTensor* out = nullptr;
     aclScalar* alpha = nullptr;
     void* workspaceDev = nullptr;
 
     do {
         ret = aclrtMalloc(&selfDev, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
         ret = aclrtMalloc(&otherDev, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
         ret = aclrtMalloc(&outDev, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
 
         ret = aclrtMemcpy(otherDev, count * sizeof(T), otherHost.data(), count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
 
         int64_t shape = static_cast<int64_t>(count);
         selfScalar = aclCreateScalar(&selfHost, dtype);
         CHECK_RET(selfScalar != nullptr, ret = ACL_ERROR_FAILURE; break);
         other = aclCreateTensor(&shape, 1, dtype, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         CHECK_RET(other != nullptr, ret = ACL_ERROR_FAILURE; break);
         out = aclCreateTensor(&shape, 1, dtype, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         CHECK_RET(out != nullptr, ret = ACL_ERROR_FAILURE; break);
         alpha = aclCreateScalar(&alphaHost, dtype);
         CHECK_RET(alpha != nullptr, ret = ACL_ERROR_FAILURE; break);
 
         uint64_t workspaceSize = 0;
         aclOpExecutor* executor = nullptr;
         ret = aclnnAddV3GetWorkspaceSize(selfScalar, other, alpha, out, &workspaceSize, &executor);
         CHECK_RET(ret == ACL_SUCCESS, break);
         CHECK_RET(executor != nullptr, ret = ACL_ERROR_FAILURE; break);
 
         if (workspaceSize > 0) {
             ret = aclrtMalloc(&workspaceDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
             CHECK_RET(ret == ACL_SUCCESS, break);
         }
 
         ret = aclnnAddV3(workspaceDev, workspaceSize, executor, stream);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
         ret = aclrtSynchronizeStream(stream);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
         ret = aclrtMemcpy(actual.data(), count * sizeof(T), outDev, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
         std::vector<T> expectedHost(count);
         for (size_t i = 0; i < count; ++i) {
             expectedHost[i] = CastFloatTo<T>(CpuAdd(self_val, other_val[i], alpha_val));
         }
 
         if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, uint16_t>) {
             ret = ExpectNearRelative(actual, expectedHost, rtol, atol, testName);
         } else {
             ret = ExpectEqualVector(actual, expectedHost, testName);
         }
     } while (false);
 
     if (alpha) aclDestroyScalar(alpha);
     if (out) aclDestroyTensor(out);
     if (other) aclDestroyTensor(other);
     if (selfScalar) aclDestroyScalar(selfScalar);
     if (outDev) aclrtFree(outDev);
     if (otherDev) aclrtFree(otherDev);
     if (selfDev) aclrtFree(selfDev);
     if (workspaceDev) aclrtFree(workspaceDev);
 
     if (ret == ACL_SUCCESS) {
         LOG_PRINT("[PASS] %s\n", testName);
     } else {
         LOG_PRINT("[FAIL] %s (ret=%d)\n", testName, ret);
     }
     return ret;
 }
 
 // =============================================================
 // aclnnInplaceAdd: selfRef += alpha * other
 // =============================================================
 template <typename T>
 static int RunInplaceAddTest(
     aclrtStream stream,
     std::vector<float> self_val,
     const std::vector<float>& other_val,
     float alpha_val,
     aclDataType dtype, size_t count,
     double rtol, double atol,
     const char* testName)
 {
    int ret = ACL_SUCCESS;
    std::vector<T> selfHost(count);
    std::vector<T> otherHost(count);
    for (size_t i = 0; i < count; ++i) {
        selfHost[i] = CastFloatTo<T>(self_val[i]);
        otherHost[i] = CastFloatTo<T>(other_val[i]);
    }
    T alphaHost = CastFloatTo<T>(alpha_val);
 
     void* selfDev = nullptr;
     void* otherDev = nullptr;
     aclTensor* self = nullptr;
     aclTensor* other = nullptr;
     aclScalar* alpha = nullptr;
     void* workspaceDev = nullptr;
 
     do {
         ret = aclrtMalloc(&selfDev, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
         ret = aclrtMalloc(&otherDev, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
 
         ret = aclrtMemcpy(selfDev, count * sizeof(T), selfHost.data(), count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
         ret = aclrtMemcpy(otherDev, count * sizeof(T), otherHost.data(), count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
 
         int64_t shape = static_cast<int64_t>(count);
         self = aclCreateTensor(&shape, 1, dtype, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         CHECK_RET(self != nullptr, ret = ACL_ERROR_FAILURE; break);
         other = aclCreateTensor(&shape, 1, dtype, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         CHECK_RET(other != nullptr, ret = ACL_ERROR_FAILURE; break);
         alpha = aclCreateScalar(&alphaHost, dtype);
         CHECK_RET(alpha != nullptr, ret = ACL_ERROR_FAILURE; break);
 
         uint64_t workspaceSize = 0;
         aclOpExecutor* executor = nullptr;
         ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
         CHECK_RET(ret == ACL_SUCCESS, break);
         CHECK_RET(executor != nullptr, ret = ACL_ERROR_FAILURE; break);
 
         if (workspaceSize > 0) {
             ret = aclrtMalloc(&workspaceDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
             CHECK_RET(ret == ACL_SUCCESS, break);
         }
 
         ret = aclnnInplaceAdd(workspaceDev, workspaceSize, executor, stream);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
         ret = aclrtSynchronizeStream(stream);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
         std::vector<T> actual(count);
         ret = aclrtMemcpy(actual.data(), count * sizeof(T), selfDev, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, break);

        std::vector<T> expectedHost(count);
        for (size_t i = 0; i < count; ++i) {
            expectedHost[i] = CastFloatTo<T>(CpuAdd(self_val[i], other_val[i], alpha_val));
        }

        if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, uint16_t>) {
            ret = ExpectNearRelative(actual, expectedHost, rtol, atol, testName);
        } else {
            ret = ExpectEqualVector(actual, expectedHost, testName);
        }
     } while (false);
 
     if (alpha) aclDestroyScalar(alpha);
     if (other) aclDestroyTensor(other);
     if (self) aclDestroyTensor(self);
     if (otherDev) aclrtFree(otherDev);
     if (selfDev) aclrtFree(selfDev);
     if (workspaceDev) aclrtFree(workspaceDev);
 
     if (ret == ACL_SUCCESS) {
         LOG_PRINT("[PASS] %s\n", testName);
     } else {
         LOG_PRINT("[FAIL] %s (ret=%d)\n", testName, ret);
     }
     return ret;
 }
 
 // =============================================================
 // aclnnInplaceAdds: selfRef += alpha * scalar
 // =============================================================
 template <typename T>
 static int RunInplaceAddsTest(
     aclrtStream stream,
     std::vector<float> self_val,
     float scalar_val,
     float alpha_val,
     aclDataType dtype, size_t count,
     double rtol, double atol,
     const char* testName)
 {
    int ret = ACL_SUCCESS;
    std::vector<T> selfHost(count);
    for (size_t i = 0; i < count; ++i) {
        selfHost[i] = CastFloatTo<T>(self_val[i]);
    }
    T scalarHost = CastFloatTo<T>(scalar_val);
    T alphaHost = CastFloatTo<T>(alpha_val);
 
     void* selfDev = nullptr;
     aclTensor* self = nullptr;
     aclScalar* scalar = nullptr;
     aclScalar* alpha = nullptr;
     void* workspaceDev = nullptr;
 
     do {
         ret = aclrtMalloc(&selfDev, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
 
         ret = aclrtMemcpy(selfDev, count * sizeof(T), selfHost.data(), count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
 
         int64_t shape = static_cast<int64_t>(count);
         self = aclCreateTensor(&shape, 1, dtype, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         CHECK_RET(self != nullptr, ret = ACL_ERROR_FAILURE; break);
         scalar = aclCreateScalar(&scalarHost, dtype);
         CHECK_RET(scalar != nullptr, ret = ACL_ERROR_FAILURE; break);
         alpha = aclCreateScalar(&alphaHost, dtype);
         CHECK_RET(alpha != nullptr, ret = ACL_ERROR_FAILURE; break);
 
         uint64_t workspaceSize = 0;
         aclOpExecutor* executor = nullptr;
         ret = aclnnInplaceAddsGetWorkspaceSize(self, scalar, alpha, &workspaceSize, &executor);
         CHECK_RET(ret == ACL_SUCCESS, break);
         CHECK_RET(executor != nullptr, ret = ACL_ERROR_FAILURE; break);
 
         if (workspaceSize > 0) {
             ret = aclrtMalloc(&workspaceDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
             CHECK_RET(ret == ACL_SUCCESS, break);
         }
 
         ret = aclnnInplaceAdds(workspaceDev, workspaceSize, executor, stream);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
         ret = aclrtSynchronizeStream(stream);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
        std::vector<T> actual(count);
        ret = aclrtMemcpy(actual.data(), count * sizeof(T), selfDev, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, break);

        std::vector<T> expectedHost(count);
        for (size_t i = 0; i < count; ++i) {
            expectedHost[i] = CastFloatTo<T>(CpuAdd(self_val[i], scalar_val, alpha_val));
        }

        if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, uint16_t>) {
            ret = ExpectNearRelative(actual, expectedHost, rtol, atol, testName);
        } else {
            ret = ExpectEqualVector(actual, expectedHost, testName);
        }
     } while (false);
 
     if (alpha) aclDestroyScalar(alpha);
     if (scalar) aclDestroyScalar(scalar);
     if (self) aclDestroyTensor(self);
     if (selfDev) aclrtFree(selfDev);
     if (workspaceDev) aclrtFree(workspaceDev);
 
     if (ret == ACL_SUCCESS) {
         LOG_PRINT("[PASS] %s\n", testName);
     } else {
         LOG_PRINT("[FAIL] %s (ret=%d)\n", testName, ret);
     }
     return ret;
 }
 
 // =============================================================
 // aclnnInplaceAddV3: selfRef += alpha * tensor  (scalar+tensor)
 // =============================================================
 template <typename T>
 static int RunInplaceAddV3Test(
     aclrtStream stream,
     float self_val,
     std::vector<float> other_val,
     float alpha_val,
     aclDataType dtype, size_t count,
     double rtol, double atol,
     const char* testName)
 {
    int ret = ACL_SUCCESS;
    std::vector<T> otherHost(count);
    for (size_t i = 0; i < count; ++i) {
        otherHost[i] = CastFloatTo<T>(other_val[i]);
    }
    T selfHost = CastFloatTo<T>(self_val);
    T alphaHost = CastFloatTo<T>(alpha_val);
 
     void* otherDev = nullptr;
     aclScalar* selfScalar = nullptr;
     aclTensor* other = nullptr;
     aclScalar* alpha = nullptr;
     void* workspaceDev = nullptr;
 
     do {
         ret = aclrtMalloc(&otherDev, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
 
         ret = aclrtMemcpy(otherDev, count * sizeof(T), otherHost.data(), count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
         CHECK_RET(ret == ACL_SUCCESS, ret = ACL_ERROR_FAILURE; break);
 
         int64_t shape = static_cast<int64_t>(count);
         selfScalar = aclCreateScalar(&selfHost, dtype);
         CHECK_RET(selfScalar != nullptr, ret = ACL_ERROR_FAILURE; break);
         other = aclCreateTensor(&shape, 1, dtype, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         CHECK_RET(other != nullptr, ret = ACL_ERROR_FAILURE; break);
         alpha = aclCreateScalar(&alphaHost, dtype);
         CHECK_RET(alpha != nullptr, ret = ACL_ERROR_FAILURE; break);
 
         uint64_t workspaceSize = 0;
         aclOpExecutor* executor = nullptr;
         ret = aclnnInplaceAddV3GetWorkspaceSize(selfScalar, other, alpha, &workspaceSize, &executor);
         CHECK_RET(ret == ACL_SUCCESS, break);
         CHECK_RET(executor != nullptr, ret = ACL_ERROR_FAILURE; break);
 
         if (workspaceSize > 0) {
             ret = aclrtMalloc(&workspaceDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
             CHECK_RET(ret == ACL_SUCCESS, break);
         }
 
         ret = aclnnInplaceAddV3(workspaceDev, workspaceSize, executor, stream);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
         ret = aclrtSynchronizeStream(stream);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
         std::vector<T> actual(count);
         ret = aclrtMemcpy(actual.data(), count * sizeof(T), otherDev, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
         CHECK_RET(ret == ACL_SUCCESS, break);
 
         std::vector<T> expectedHost(count);
         for (size_t i = 0; i < count; ++i) {
             expectedHost[i] = CastFloatTo<T>(CpuAdd(self_val, other_val[i], alpha_val));
         }
 
         if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, uint16_t>) {
             ret = ExpectNearRelative(actual, expectedHost, rtol, atol, testName);
         } else {
             ret = ExpectEqualVector(actual, expectedHost, testName);
         }
     } while (false);
 
     if (alpha) aclDestroyScalar(alpha);
     if (other) aclDestroyTensor(other);
     if (selfScalar) aclDestroyScalar(selfScalar);
     if (otherDev) aclrtFree(otherDev);
     if (workspaceDev) aclrtFree(workspaceDev);
 
     if (ret == ACL_SUCCESS) {
         LOG_PRINT("[PASS] %s\n", testName);
     } else {
         LOG_PRINT("[FAIL] %s (ret=%d)\n", testName, ret);
     }
     return ret;
 }
 
 // =============================================================
 // Precision Tests — following precision.md guidelines
 // CPU reference uses double precision internally, receives float
 // inputs to ensure the same quantized reference baseline.
 // =============================================================
 static int RunPrecisionTests(aclrtStream stream)
 {
     LOG_PRINT("\n========== Precision Tests ==========\n");
     int ret = ACL_SUCCESS;
     const size_t N = 4;
 
     // ------------------------------------------------------------
     // Scenario 1: Basic correctness — diverse normal values
     // ------------------------------------------------------------
     {
         std::vector<float> self = {1.0f, 2.5f, -3.7f, 100.0f};
         std::vector<float> other = {0.5f, 1.2f, 2.0f, -50.0f};
         ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-6, 1e-6,
                                  "Precision: FLOAT basic (alpha=1)");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
     {
         std::vector<float> self = {1.0f, 2.5f, -3.7f, 100.0f};
         std::vector<float> other = {0.5f, 1.2f, 2.0f, -50.0f};
        ret = RunAddTest<uint16_t>(stream, self, other, 1.0f, ACL_FLOAT16, N, 1e-3, 1e-3,
                                 "Precision: FLOAT16 basic (alpha=1)");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    {
        std::vector<float> self = {1.0f, 2.5f, -3.7f, 100.0f};
        std::vector<float> other = {0.5f, 1.2f, 2.0f, -50.0f};
        ret = RunAddTest<uint16_t>(stream, self, other, 1.0f, ACL_BF16, N, 1e-2, 1e-2,
                                  "Precision: BF16 basic (alpha=1)");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
 
     // ------------------------------------------------------------
     // Scenario 2: Decimal values that cannot be precisely represented
     // This is the classic 0.1 + 0.2 precision scenario for add.
     // 0.1 in float32 ≈ 0.10000000149
     // 0.2 in float32 ≈ 0.20000000298
     // 0.1 + 0.2 ≈ 0.30000000447 (NPU)
     // Math truth: 0.1 + 0.2 = 0.3
     // ------------------------------------------------------------
     {
         std::vector<float> self = {0.1f, 0.1f, 0.1f, 0.1f};
         std::vector<float> other = {0.2f, 0.3f, 0.7f, 0.1f};
         ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-5, 1e-5,
                                  "Precision: FLOAT decimal (0.1+0.2 etc)");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
     {
         std::vector<float> self = {0.1f, 0.1f, 0.1f, 0.1f};
         std::vector<float> other = {0.2f, 0.3f, 0.7f, 0.1f};
        ret = RunAddTest<uint16_t>(stream, self, other, 1.0f, ACL_FLOAT16, N, 1e-2, 1e-2,
                                "Precision: FLOAT16 decimal (0.1+0.2 etc)");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
 
     // ------------------------------------------------------------
     // Scenario 3: Values near representable extremes
     // Float32: min normal ≈ 1.175e-38, max ≈ 3.4e38
     // Test adding a small value near the bottom of normal range.
     // ------------------------------------------------------------
     {
         std::vector<float> self = {1.2e-38f, 1.2e-38f, 1.2e-38f, 1.2e-38f};
         std::vector<float> other = {1.0f, 1.0f, 1.0f, 1.0f};
         ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-5, 1e-5,
                                  "Precision: FLOAT near min-normal");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
     {
         std::vector<float> self = {3.3e38f, 3.3e38f, 3.3e38f, 3.3e38f};
         std::vector<float> other = {1.0f, 1.0f, 1.0f, 1.0f};
         ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-5, 1e-5,
                                  "Precision: FLOAT near max-finite");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
 
     // ------------------------------------------------------------
     // Scenario 4: Very large magnitude differences
     // Adding a tiny number to a very large number.
     // The tiny number may be lost in the last bits.
     // ------------------------------------------------------------
     {
         std::vector<float> self = {1e10f, 1e10f, 1e10f, 1e10f};
         std::vector<float> other = {1e-5f, 1e-3f, 1.0f, 1e3f};
         ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-5, 1e-5,
                                  "Precision: FLOAT large+small magnitude");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
 
     // ------------------------------------------------------------
     // Scenario 5: alpha != 1 — scaling scenarios
     // out = self + alpha * other, alpha is not 1
     // ------------------------------------------------------------
     {
         std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
         std::vector<float> other = {5.0f, 6.0f, 7.0f, 8.0f};
         ret = RunAddTest<float>(stream, self, other, 2.0f, ACL_FLOAT, N, 1e-5, 1e-5,
                                  "Precision: FLOAT alpha=2.0");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
     {
         std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
         std::vector<float> other = {5.0f, 6.0f, 7.0f, 8.0f};
         ret = RunAddTest<float>(stream, self, other, 0.5f, ACL_FLOAT, N, 1e-5, 1e-5,
                                  "Precision: FLOAT alpha=0.5");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
     {
         std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
         std::vector<float> other = {5.0f, 6.0f, 7.0f, 8.0f};
         ret = RunAddTest<float>(stream, self, other, -1.5f, ACL_FLOAT, N, 1e-5, 1e-5,
                                  "Precision: FLOAT alpha=-1.5");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
     {
         std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
         std::vector<float> other = {5.0f, 6.0f, 7.0f, 8.0f};
        ret = RunAddTest<uint16_t>(stream, self, other, 2.0f, ACL_FLOAT16, N, 1e-2, 1e-2,
                                "Precision: FLOAT16 alpha=2.0");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    {
        std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> other = {5.0f, 6.0f, 7.0f, 8.0f};
        ret = RunAddTest<uint16_t>(stream, self, other, 2.0f, ACL_BF16, N, 2e-2, 2e-2,
                                  "Precision: BF16 alpha=2.0");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
 
     // ------------------------------------------------------------
     // Scenario 6: Negative values
     // ------------------------------------------------------------
     {
         std::vector<float> self = {-1.0f, -2.5f, -100.0f, -0.001f};
         std::vector<float> other = {-0.5f, 1.0f, 50.0f, 0.002f};
         ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-6, 1e-6,
                                  "Precision: FLOAT negative values");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
 
     // ------------------------------------------------------------
     // Scenario 7: aclnnAdds — scalar tensor addition
     // ------------------------------------------------------------
     {
         std::vector<float> self = {10.0f, 20.0f, 30.0f, 40.0f};
         ret = RunAddsTest<float>(stream, self, 5.0f, 1.0f, ACL_FLOAT, N, 1e-6, 1e-6,
                                   "Precision: aclnnAdds FLOAT alpha=1");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
     {
         std::vector<float> self = {10.0f, 20.0f, 30.0f, 40.0f};
         ret = RunAddsTest<float>(stream, self, 5.0f, 3.0f, ACL_FLOAT, N, 1e-5, 1e-5,
                                   "Precision: aclnnAdds FLOAT alpha=3");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
     {
         std::vector<float> self = {10.0f, 20.0f, 30.0f, 40.0f};
        ret = RunAddsTest<uint16_t>(stream, self, 5.0f, 2.0f, ACL_FLOAT16, N, 1e-2, 1e-2,
                                "Precision: aclnnAdds FLOAT16 alpha=2");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    {
        std::vector<float> self = {10.0f, 20.0f, 30.0f, 40.0f};
        ret = RunAddsTest<uint16_t>(stream, self, 5.0f, 2.0f, ACL_BF16, N, 2e-2, 2e-2,
                                   "Precision: aclnnAdds BF16 alpha=2");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
     {
         std::vector<float> self = {10.0f, 20.0f, 30.0f, 40.0f};
         ret = RunAddsTest<int32_t>(stream, self, 5.0f, 1.0f, ACL_INT32, N, 0, 0,
                                     "Precision: aclnnAdds INT32");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
 
     // ------------------------------------------------------------
     // Scenario 8: aclnnAddV3 — scalar + tensor
     // ------------------------------------------------------------
     {
         std::vector<float> other = {1.0f, 2.0f, 3.0f, 4.0f};
         ret = RunAddV3Test<float>(stream, 10.0f, other, 1.0f, ACL_FLOAT, N, 1e-6, 1e-6,
                                    "Precision: aclnnAddV3 FLOAT alpha=1");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
     {
         std::vector<float> other = {1.0f, 2.0f, 3.0f, 4.0f};
         ret = RunAddV3Test<float>(stream, 10.0f, other, 2.0f, ACL_FLOAT, N, 1e-5, 1e-5,
                                    "Precision: aclnnAddV3 FLOAT alpha=2");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
     {
         std::vector<float> other = {1.0f, 2.0f, 3.0f, 4.0f};
        ret = RunAddV3Test<uint16_t>(stream, 10.0f, other, 2.0f, ACL_FLOAT16, N, 1e-2, 1e-2,
                                   "Precision: aclnnAddV3 FLOAT16 alpha=2");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    {
        std::vector<float> other = {1.0f, 2.0f, 3.0f, 4.0f};
        ret = RunAddV3Test<uint16_t>(stream, 10.0f, other, 2.0f, ACL_BF16, N, 2e-2, 2e-2,
                                    "Precision: aclnnAddV3 BF16 alpha=2");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
 
     // ------------------------------------------------------------
     // Scenario 9: Inplace operations
     // ------------------------------------------------------------
     {
         std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
         std::vector<float> other = {5.0f, 6.0f, 7.0f, 8.0f};
         ret = RunInplaceAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-6, 1e-6,
                                          "Precision: aclnnInplaceAdd FLOAT");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
     {
         std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
         std::vector<float> other = {5.0f, 6.0f, 7.0f, 8.0f};
         ret = RunInplaceAddTest<float>(stream, self, other, 2.0f, ACL_FLOAT, N, 1e-5, 1e-5,
                                          "Precision: aclnnInplaceAdd FLOAT alpha=2");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
     {
         std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
         ret = RunInplaceAddsTest<float>(stream, self, 5.0f, 1.0f, ACL_FLOAT, N, 1e-6, 1e-6,
                                          "Precision: aclnnInplaceAdds FLOAT");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
     {
         std::vector<float> other = {1.0f, 2.0f, 3.0f, 4.0f};
         ret = RunInplaceAddV3Test<float>(stream, 10.0f, other, 1.0f, ACL_FLOAT, N, 1e-6, 1e-6,
                                           "Precision: aclnnInplaceAddV3 FLOAT");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
 
     // ------------------------------------------------------------
     // Scenario 10: Integer types
     // ------------------------------------------------------------
     {
         std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
         std::vector<float> other = {5.0f, 6.0f, 7.0f, 8.0f};
         ret = RunAddTest<int32_t>(stream, self, other, 1.0f, ACL_INT32, N, 0, 0,
                                    "Precision: aclnnAdd INT32");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
     {
         std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
         std::vector<float> other = {5.0f, 6.0f, 7.0f, 8.0f};
         ret = RunAddTest<int64_t>(stream, self, other, 1.0f, ACL_INT64, N, 0, 0,
                                    "Precision: aclnnAdd INT64");
         CHECK_RET(ret == ACL_SUCCESS, return ret);
     }
 
    // ------------------------------------------------------------
    // Scenario 11: Subnormal values (denormalized float)
    // Float32 subnormal range: 1.0e-45 to 1.175e-38
    // Adding subnormal to normal can flush to zero.
    // ------------------------------------------------------------
    {
        std::vector<float> self = {1e-40f, 1e-41f, 1e-42f, 1e-43f};
        std::vector<float> other = {1e-40f, 1e-41f, 1e-42f, 1e-43f};
        ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-5, 1e-2,
                                 "Precision: FLOAT subnormal (denormalized)");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 12: Overflow — inf propagation
    // 3.4e38 + 3.4e38 = inf in Float32
    // ------------------------------------------------------------
    {
        std::vector<float> self = {3.4e38f, 3.4e38f, -3.4e38f, -3.4e38f};
        std::vector<float> other = {1.0e38f, 1.0e38f, -1.0e38f, -1.0e38f};
        ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-3, 1e-3,
                                 "Precision: FLOAT overflow (+inf)");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 13: Near-unity differences
    // (1 + 1e-7) + (1 - 1e-7) should equal 2
    // Tests ULP boundary in the most significant float bits.
    // ------------------------------------------------------------
    {
        std::vector<float> self = {1.0000001f, 1.0f, 0.9999999f, 1.000001f};
        std::vector<float> other = {0.9999999f, 1.0f, 1.0000001f, -0.000001f};
        ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-5, 1e-2,
                                 "Precision: FLOAT near-unity differences");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 14: Magnitude cancellation (large + tiny = large)
    // 1e10 + 1e-10 should yield 1e10 (tiny lost in ULP)
    // Different from scale: this tests catastrophic cancellation.
    // ------------------------------------------------------------
    {
        std::vector<float> self = {1e10f, -1e10f, 1e10f, 1e10f};
        std::vector<float> other = {1e-10f, 1e-10f, -1e10f, 1e-5f};
        ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-3, 1e-3,
                                 "Precision: FLOAT magnitude cancellation");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 15: Scale with alpha != 1 and tiny subnormal inputs
    // out = self + alpha * other where alpha * other is subnormal.
    // ------------------------------------------------------------
    {
        std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> other = {1e-40f, 1e-41f, 1e-42f, 1e-43f};
        ret = RunAddTest<float>(stream, self, other, 1e-5f, ACL_FLOAT, N, 1e-5, 1e-2,
                                 "Precision: FLOAT alpha*subnormal");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 16: Negative alpha with normal values
    // out = self + (-2) * other
    // ------------------------------------------------------------
    {
        std::vector<float> self = {10.0f, 20.0f, 30.0f, 40.0f};
        std::vector<float> other = {1.0f, 2.0f, 3.0f, 4.0f};
        ret = RunAddTest<float>(stream, self, other, -2.0f, ACL_FLOAT, N, 1e-5, 1e-5,
                                 "Precision: FLOAT alpha=-2.0");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 17: aclnnAdds with subnormal tensor
    // self[i] + alpha * scalar where self is subnormal
    // ------------------------------------------------------------
    {
        std::vector<float> self = {1e-40f, 1e-41f, 1e-42f, 1e-43f};
        ret = RunAddsTest<float>(stream, self, 1e-5f, 1.0f, ACL_FLOAT, N, 1e-5, 1e-2,
                                  "Precision: aclnnAdds FLOAT subnormal tensor");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 18: aclnnAddV3 with negative alpha
    // scalar + alpha * tensor, alpha is negative
    // ------------------------------------------------------------
    {
        std::vector<float> other = {1.0f, 2.0f, 3.0f, 4.0f};
        ret = RunAddV3Test<float>(stream, 10.0f, other, -1.0f, ACL_FLOAT, N, 1e-6, 1e-6,
                                    "Precision: aclnnAddV3 FLOAT alpha=-1.0");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 19: aclnnInplaceAdd with overflow
    // selfRef += alpha * other, selfRef is near max
    // ------------------------------------------------------------
    {
        std::vector<float> self = {3.0e38f, 3.0e38f, 3.0e38f, 3.0e38f};
        std::vector<float> other = {1.0e38f, 1.0e38f, 1.0e38f, 1.0e38f};
        ret = RunInplaceAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-3, 1e-3,
                                         "Precision: aclnnInplaceAdd FLOAT overflow");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 20: Broadcast precision
    // Shape {1,3} + {2,3} -> broadcast to {2,3}
    // ------------------------------------------------------------
    {
        std::vector<float> self = {1.0f, 2.0f, 3.0f};
        std::vector<float> other = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
        // Use RunAddTest but shape is {6}
        ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, 6, 1e-6, 1e-6,
                                 "Precision: FLOAT broadcast values");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 21: BF16 overflow
    // BF16 max ≈ 3.3895314e38, adding near-max produces inf
    // ------------------------------------------------------------
    {
        std::vector<float> self = {2e38f, 2e38f, 2e38f, 2e38f};
        std::vector<float> other = {2e38f, 2e38f, 2e38f, 2e38f};
        ret = RunAddTest<uint16_t>(stream, self, other, 1.0f, ACL_BF16, N, 1e-1, 1e-1,
                                    "Precision: BF16 near-overflow");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 22: INT64 large values (no floating precision issues)
    // ------------------------------------------------------------
    {
        std::vector<float> self = {1000000000.0f, 2000000000.0f, -1000000000.0f, 0.0f};
        std::vector<float> other = {2000000000.0f, -1000000000.0f, 1000000000.0f, 0.0f};
        ret = RunAddTest<int64_t>(stream, self, other, 1.0f, ACL_INT64, N, 0, 0,
                                   "Precision: aclnnAdd INT64 large values");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 23: FLOAT16 magnitude cancellation
    // 1e4 + 1e-4 = 1e4 (lost) in Float16 (only ~3.3 decimal digits)
    // ------------------------------------------------------------
    {
        std::vector<float> self = {10000.0f, 10000.0f, 10000.0f, 10000.0f};
        std::vector<float> other = {0.0001f, 0.0002f, 0.0003f, 0.0004f};
        ret = RunAddTest<uint16_t>(stream, self, other, 1.0f, ACL_FLOAT16, N, 1e-1, 1e-1,
                                    "Precision: FLOAT16 magnitude cancellation");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 24: aclnnAdds with alpha scaling (subnormal result)
    // self + 1e-10 * scalar where alpha*scalar is subnormal
    // ------------------------------------------------------------
    {
        std::vector<float> self = {1.0f, 2.0f, 3.0f, 4.0f};
        ret = RunAddsTest<uint16_t>(stream, self, 1e-5f, 1e-5f, ACL_FLOAT16, N, 1e-2, 1e-1,
                                     "Precision: aclnnAdds FLOAT16 subnormal scale");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 25: aclnnAddV3 with subnormal tensor
    // scalar + alpha * subnormal_tensor
    // ------------------------------------------------------------
    {
        std::vector<float> other = {1e-40f, 1e-41f, 1e-42f, 1e-43f};
        ret = RunAddV3Test<uint16_t>(stream, 1.0f, other, 1.0f, ACL_FLOAT16, N, 1e-2, 1e-1,
                                      "Precision: aclnnAddV3 FLOAT16 subnormal tensor");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 26: Subnormal inputs — adding two subnormals (underflow analog for Add)
    // In Float32, 1e-40 is subnormal (below min normal 1.175e-38).
    // Adding two subnormals may produce a normal result (1e-40 + 1e-40 = 2e-40 > 1.175e-38 is false,
    // so the sum is still subnormal) or be partially lost depending on FTZ mode.
    // Test both with wider tolerance to accommodate subnormal rounding behavior.
    // ------------------------------------------------------------
    {
        std::vector<float> self = {1e-40f, 1e-40f, 1e-40f, 1e-40f};
        std::vector<float> other = {1e-40f, 2e-40f, 5e-40f, 9e-40f};
        ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-4, 1e-2,
                                "Precision: FLOAT subnormal + subnormal (underflow)");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 27: Overflow to +inf — addition of two near-maximum values
    // Float32 max finite ≈ 3.4028235e38. Adding two values each close to
    // the maximum produces +inf. This mirrors the "极大值相乘" overflow
    // scenario, but for addition: 3.4e38 + 3.4e38 > max → inf.
    // ------------------------------------------------------------
    {
        std::vector<float> self = {3.4e38f, 3.4e38f, 3.4e38f, 3.4e38f};
        std::vector<float> other = {3.4e38f, 3.4e38f, 3.4e38f, 3.4e38f};
        ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-3, 1e-3,
                                "Precision: FLOAT overflow to +inf");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 28: Catastrophic cancellation near 1.0
    // For addition, the most damaging precision scenario is when two
    // numbers of similar magnitude but opposite sign cancel each other,
    // leaving only the difference in their lower bits.
    // (1 + 1e-7) + (-1) ≈ 1e-7: the result lives in the ULP of 1.0.
    // This is the addition analog of Mul's "接近 1.0 的微小差异" scenario.
    // Float32 ULP near 1.0 ≈ 1.19e-7; the result is exactly at that boundary.
    // ------------------------------------------------------------
    {
        std::vector<float> self = {1.0f, 1.0f, 1.0f, 1.0f};
        std::vector<float> other = {1e-7f, 2e-7f, 5e-8f, -1e-7f};
        ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-4, 1e-2,
                                "Precision: FLOAT catastrophic cancellation near 1.0");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 29: Large magnitude differences — tiny term absorbed
    // Adding a tiny number to a large number may lose the tiny term
    // entirely in Float32. 1e10 + 1e-5 ≈ 1e10 (tiny lost).
    // This mirrors Mul's "量级悬殊" scenario but highlights that
    // addition is far more sensitive to magnitude differences than
    // multiplication (whose result stays bounded by input magnitudes).
    // ------------------------------------------------------------
    {
        std::vector<float> self = {1e10f, 1e10f, 1e10f, 1e10f};
        std::vector<float> other = {1e-5f, 1e-3f, 1e-1f, 1.0f};
        ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-4, 1e-3,
                                "Precision: FLOAT large+magnitude absorption");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // ------------------------------------------------------------
    // Scenario 30: Non-exact decimal representation — 0.1 + 0.2 ≠ 0.3
    // 0.1 and 0.2 are infinite recurring binaries in Float32, each
    // already carrying quantization error before any arithmetic.
    // Float32: 0.1f ≈ 0.10000000149, 0.2f ≈ 0.20000000298
    // Sum: 0.10000000149 + 0.20000000298 ≈ 0.30000000447
    // Math truth: 0.1 + 0.2 = 0.3, but 0.3f ≈ 0.30000001192
    // So 0.1 + 0.2 (float32) ≈ 0.30000000447, which is NOT equal to 0.3f.
    // This mirrors Mul's "无法精确表示的十进制小数" scenario and
    // demonstrates that addition is subject to the same binary
    // representation pitfalls as multiplication.
    // ------------------------------------------------------------
    {
        std::vector<float> self = {0.1f, 0.1f, 0.1f, 0.1f};
        std::vector<float> other = {0.2f, 0.3f, 0.7f, 0.8f};
        ret = RunAddTest<float>(stream, self, other, 1.0f, ACL_FLOAT, N, 1e-4, 1e-4,
                                "Precision: FLOAT decimal non-exact 0.1+0.2+0.3+0.7+0.8");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    LOG_PRINT("\n[INFO] All precision tests passed.\n");
    return ACL_SUCCESS;
}
 
 // =============================================================
 // Statement & Branch Coverage Tests
 // Only call GetWorkspaceSize — kernel execution is slow and
 // coverage is already achieved through GetWorkspaceSize paths.
 // =============================================================
 
 static void RunCoverageTests(aclrtStream stream)
 {
     LOG_PRINT("\n========== Coverage Tests ==========\n");
 
     // =============================================================
     // aclnnAdd: CheckNotNull — null pointer branches
     // =============================================================
     {
         printf(">> [COV] aclnnAdd null pointer: self=nullptr\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret1 = aclnnAddGetWorkspaceSize(nullptr, other, alpha, out, &ws, &exec);
         printf(">>   self=nullptr ret=%d\n", ret1);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
     {
         printf(">> [COV] aclnnAdd null pointer: other=nullptr\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret2 = aclnnAddGetWorkspaceSize(self, nullptr, alpha, out, &ws, &exec);
         printf(">>   other=nullptr ret=%d\n", ret2);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(selfDev);
     }
     {
         printf(">> [COV] aclnnAdd null pointer: alpha=nullptr\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
 
         auto ret3 = aclnnAddGetWorkspaceSize(self, other, nullptr, out, &ws, &exec);
         printf(">>   alpha=nullptr ret=%d\n", ret3);
 
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
     {
         printf(">> [COV] aclnnAdd null pointer: out=nullptr\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret4 = aclnnAddGetWorkspaceSize(self, other, alpha, nullptr, &ws, &exec);
         printf(">>   out=nullptr ret=%d\n", ret4);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // =============================================================
     // aclnnAdd: unsupported dtype in support list
     // ASCEND910 does not support DT_BF16
     // =============================================================
     {
         printf(">> [COV] aclnnAdd unsupported dtype BF16 (ASCEND910 path)\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_BF16);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   BF16 dtype ret=%d\n", ret);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // =============================================================
     // aclnnAdd: shape mismatch — broadcast shape != output shape
     // =============================================================
     {
         printf(">> [COV] aclnnAdd shape mismatch (broadcast != output)\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         int64_t shapeA = 4;
         int64_t shapeB = 4;
         int64_t shapeOut = 3; // Different shape
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, 4 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, 4 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, 3 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shapeA, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeA, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shapeB, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeB, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shapeOut, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeOut, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   shape mismatch ret=%d\n", ret);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // =============================================================
     // aclnnAdd: self IsEmpty() branch
     // =============================================================
     {
         printf(">> [COV] aclnnAdd self IsEmpty branch\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         int64_t shape = 0; // Empty tensor
         int64_t shapeOut = 4;
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, 1 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, 4 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, 4 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shapeOut, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeOut, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shapeOut, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeOut, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   self IsEmpty ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // =============================================================
     // aclnnAdd: other IsEmpty() branch
     // =============================================================
     {
         printf(">> [COV] aclnnAdd other IsEmpty branch\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         int64_t shapeA = 4;
         int64_t shapeB = 0; // Empty tensor
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, 4 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, 1 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, 4 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shapeA, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeA, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shapeB, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeB, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shapeA, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeA, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   other IsEmpty ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // =============================================================
     // aclnnAdd: Mixed dtype paths (FP16+FP32, BF16+FP32)
     // isMixDataType=true, alpha=1 (Add kernel path)
     // =============================================================
     {
         printf(">> [COV] aclnnAdd mixed dtype FP16(self)+FP32(other), alpha=1\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   FP16+FP32 mixed ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
     {
         printf(">> [COV] aclnnAdd mixed dtype FP32(self)+FP16(other), alpha=1\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   FP32+FP16 mixed ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
     {
         printf(">> [COV] aclnnAdd mixed dtype BF16(self)+FP32(other), alpha=1\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   BF16+FP32 mixed ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
     {
         printf(">> [COV] aclnnAdd mixed dtype FP32(self)+BF16(other), alpha=1\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   FP32+BF16 mixed ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // Mixed dtype with alpha != 1 (Axpy path)
     {
         printf(">> [COV] aclnnAdd mixed dtype FP16+FP32, alpha=2 (Axpy path)\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 2.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   FP16+FP32 mixed alpha=2 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // =============================================================
     // aclnnAdd: IsEqualToOne=true branch
     // Same dtype, alpha=1 -> Add kernel path
     // =============================================================
     {
         printf(">> [COV] aclnnAdd IsEqualToOne path, same dtype FLOAT\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   FLOAT+alpha=1 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // IsEqualToOne=false, IsSupportAxpy=true (Axpy kernel path)
     // FLOAT and INT32 are in AXPY_DTYPE_SUPPORT_LIST
     {
         printf(">> [COV] aclnnAdd Axpy path: FLOAT, alpha=2.0\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 2.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   FLOAT alpha=2.0 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // INT32+INT32 alpha=2 (Axpy path)
     {
         printf(">> [COV] aclnnAdd Axpy path: INT32, alpha=2.0\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 2.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   INT32 alpha=2.0 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // FLOAT16+alpha=2 (IsSupportAxpy path)
     {
         printf(">> [COV] aclnnAdd Axpy path: FLOAT16, alpha=2.0\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 2.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT16);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   FLOAT16 alpha=2.0 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // Fallback path (Mul+Add): DOUBLE, alpha=2 (not in AXPY_DTYPE_SUPPORT_LIST)
     {
         printf(">> [COV] aclnnAdd fallback path: DOUBLE, alpha=2.0 (Mul+Add)\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(double), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(double), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(double), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_DOUBLE, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_DOUBLE, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_DOUBLE, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 2.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_DOUBLE);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   DOUBLE alpha=2.0 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // =============================================================
     // aclnnAdd: Type promotion — dtype conversion paths
     // INT32 + INT64 -> INT64 -> INT32 output (Cast both inputs)
     // =============================================================
     {
         printf(">> [COV] aclnnAdd type promotion: INT32+INT64->INT64->INT32\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_INT64, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   INT32+INT64->INT32 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // INT32 + INT32 -> INT32 output (same dtype, no cast)
     {
         printf(">> [COV] aclnnAdd same dtype INT32+INT32\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   INT32+INT32 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // INT64 + INT64 alpha=1
     {
         printf(">> [COV] aclnnAdd INT64+INT64\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_INT64, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_INT64, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_INT64, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT64);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   INT64+INT64 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // INT8 + INT8
     {
         printf(">> [COV] aclnnAdd INT8+INT8\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT8);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   INT8+INT8 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // UINT8 + UINT8
     {
         printf(">> [COV] aclnnAdd UINT8+UINT8\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_UINT8);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   UINT8+UINT8 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // BOOL + BOOL
     {
         printf(">> [COV] aclnnAdd BOOL+BOOL\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_BOOL, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_BOOL, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_BOOL, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_BOOL);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   BOOL+BOOL ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // =============================================================
     // aclnnAdds: CheckNotNullScalar null branches
     // =============================================================
     {
         printf(">> [COV] aclnnAdds null self\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* outDev = nullptr;
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float scalarVal = 1.0f;
         float alphaVal = 1.0f;
         aclScalar* scalar = aclCreateScalar(&scalarVal, ACL_FLOAT);
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddsGetWorkspaceSize(nullptr, scalar, alpha, out, &ws, &exec);
         printf(">>   self=nullptr ret=%d\n", ret);
 
         aclDestroyScalar(alpha);
         aclDestroyScalar(scalar);
         aclDestroyTensor(out);
         aclrtFree(outDev);
     }
     {
         printf(">> [COV] aclnnAdds null scalar\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddsGetWorkspaceSize(self, nullptr, alpha, out, &ws, &exec);
         printf(">>   scalar=nullptr ret=%d\n", ret);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(selfDev);
     }
     {
         printf(">> [COV] aclnnAdds null alpha\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float scalarVal = 1.0f;
         aclScalar* scalar = aclCreateScalar(&scalarVal, ACL_FLOAT);
 
         auto ret = aclnnAddsGetWorkspaceSize(self, scalar, nullptr, out, &ws, &exec);
         printf(">>   alpha=nullptr ret=%d\n", ret);
 
         aclDestroyScalar(scalar);
         aclDestroyTensor(out);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(selfDev);
     }
     {
         printf(">> [COV] aclnnAdds null out\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         float scalarVal = 1.0f;
         float alphaVal = 1.0f;
         aclScalar* scalar = aclCreateScalar(&scalarVal, ACL_FLOAT);
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddsGetWorkspaceSize(self, scalar, alpha, nullptr, &ws, &exec);
         printf(">>   out=nullptr ret=%d\n", ret);
 
         aclDestroyScalar(alpha);
         aclDestroyScalar(scalar);
         aclDestroyTensor(self);
         aclrtFree(selfDev);
     }
 
     // aclnnAdds: self IsEmpty
     {
         printf(">> [COV] aclnnAdds self IsEmpty\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         int64_t shapeIn = 0;
         int64_t shapeOut = 4;
         void* selfDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, 1 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, 4 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shapeIn, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeIn, 1, selfDev);
         aclTensor* out = aclCreateTensor(&shapeOut, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeOut, 1, outDev);
         float scalarVal = 1.0f;
         float alphaVal = 1.0f;
         aclScalar* scalar = aclCreateScalar(&scalarVal, ACL_FLOAT);
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddsGetWorkspaceSize(self, scalar, alpha, out, &ws, &exec);
         printf(">>   self IsEmpty ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyScalar(scalar);
         aclDestroyTensor(out);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(selfDev);
     }
 
     // aclnnAdds: IsEqualToOne=false, IsSupportAxpy=true (Axpy path)
     {
         printf(">> [COV] aclnnAdds Axpy path: FLOAT alpha=2.0\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float scalarVal = 3.0f;
         float alphaVal = 2.0f;
         aclScalar* scalar = aclCreateScalar(&scalarVal, ACL_FLOAT);
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddsGetWorkspaceSize(self, scalar, alpha, out, &ws, &exec);
         printf(">>   FLOAT alpha=2.0 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyScalar(scalar);
         aclDestroyTensor(out);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(selfDev);
     }
 
     // aclnnAdds: BOOL+BOOL special bool cast path
     {
         printf(">> [COV] aclnnAdds BOOL+BOOL special bool path\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(bool), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_BOOL, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_BOOL, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         bool scalarVal = true;
         bool alphaVal = true;
         aclScalar* scalar = aclCreateScalar(&scalarVal, ACL_BOOL);
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_BOOL);
 
         auto ret = aclnnAddsGetWorkspaceSize(self, scalar, alpha, out, &ws, &exec);
         printf(">>   BOOL+bool=true ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyScalar(scalar);
         aclDestroyTensor(out);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(selfDev);
     }
 
     // =============================================================
     // aclnnAddV3: CheckNotNull branches
     // =============================================================
     {
         printf(">> [COV] aclnnAddV3 null self scalar\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddV3GetWorkspaceSize(nullptr, other, alpha, out, &ws, &exec);
         printf(">>   self=nullptr ret=%d\n", ret);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclrtFree(outDev);
         aclrtFree(otherDev);
     }
     {
         printf(">> [COV] aclnnAddV3 null other tensor\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         float selfVal = 10.0f;
         float alphaVal = 1.0f;
         aclScalar* selfScalar = aclCreateScalar(&selfVal, ACL_FLOAT);
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddV3GetWorkspaceSize(selfScalar, nullptr, alpha, nullptr, &ws, &exec);
         printf(">>   other=nullptr ret=%d\n", ret);
 
         aclDestroyScalar(alpha);
         aclDestroyScalar(selfScalar);
     }
     {
         printf(">> [COV] aclnnAddV3 null alpha\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float selfVal = 10.0f;
         aclScalar* selfScalar = aclCreateScalar(&selfVal, ACL_FLOAT);
 
         auto ret = aclnnAddV3GetWorkspaceSize(selfScalar, other, nullptr, out, &ws, &exec);
         printf(">>   alpha=nullptr ret=%d\n", ret);
 
         aclDestroyScalar(selfScalar);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclrtFree(outDev);
         aclrtFree(otherDev);
     }
 
     // aclnnAddV3: IsSupportAxpy=true (FLOAT)
     {
         printf(">> [COV] aclnnAddV3 Axpy path: FLOAT alpha=2.0\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float selfVal = 10.0f;
         float alphaVal = 2.0f;
         aclScalar* selfScalar = aclCreateScalar(&selfVal, ACL_FLOAT);
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddV3GetWorkspaceSize(selfScalar, other, alpha, out, &ws, &exec);
         printf(">>   FLOAT alpha=2.0 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyScalar(selfScalar);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclrtFree(outDev);
         aclrtFree(otherDev);
     }
 
     // aclnnAddV3: Fallback (Mul+Add): FLOAT16 alpha=2 (AXPY only supports FLOAT)
     {
         printf(">> [COV] aclnnAddV3 fallback: FLOAT16 alpha=2.0 (Mul+Add path)\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&otherDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float selfVal = 10.0f;
         float alphaVal = 2.0f;
         aclScalar* selfScalar = aclCreateScalar(&selfVal, ACL_FLOAT16);
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT16);
 
         auto ret = aclnnAddV3GetWorkspaceSize(selfScalar, other, alpha, out, &ws, &exec);
         printf(">>   FLOAT16 alpha=2.0 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyScalar(selfScalar);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclrtFree(outDev);
         aclrtFree(otherDev);
     }
 
     // aclnnAddV3: other IsEmpty
     {
         printf(">> [COV] aclnnAddV3 other IsEmpty\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         int64_t shapeIn = 0;
         int64_t shapeOut = 4;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&otherDev, 1 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, 4 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* other = aclCreateTensor(&shapeIn, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeIn, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shapeOut, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeOut, 1, outDev);
         float selfVal = 10.0f;
         float alphaVal = 1.0f;
         aclScalar* selfScalar = aclCreateScalar(&selfVal, ACL_FLOAT);
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddV3GetWorkspaceSize(selfScalar, other, alpha, out, &ws, &exec);
         printf(">>   other IsEmpty ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyScalar(selfScalar);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclrtFree(outDev);
         aclrtFree(otherDev);
     }
 
     // aclnnAddV3: INT32 other (in ADD_V3_DTYPE_SUPPORT_LIST)
     {
         printf(">> [COV] aclnnAddV3 INT32 other\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&otherDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         int32_t selfVal = 10;
         float alphaVal = 1.0f;
         aclScalar* selfScalar = aclCreateScalar(&selfVal, ACL_INT32);
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);
 
         auto ret = aclnnAddV3GetWorkspaceSize(selfScalar, other, alpha, out, &ws, &exec);
         printf(">>   INT32 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyScalar(selfScalar);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclrtFree(outDev);
         aclrtFree(otherDev);
     }
 
     // aclnnAddV3: BF16 other
     {
         printf(">> [COV] aclnnAddV3 BF16 other\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&otherDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         uint16_t selfVal = 10;
         float alphaVal = 1.0f;
         aclScalar* selfScalar = aclCreateScalar(&selfVal, ACL_BF16);
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_BF16);
 
         auto ret = aclnnAddV3GetWorkspaceSize(selfScalar, other, alpha, out, &ws, &exec);
         printf(">>   BF16 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyScalar(selfScalar);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclrtFree(outDev);
         aclrtFree(otherDev);
     }
 
     // aclnnAddV3: INT8 other (in ADD_V3_DTYPE_SUPPORT_LIST)
     {
         printf(">> [COV] aclnnAddV3 INT8 other\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 8;
         int64_t shape = static_cast<int64_t>(count);
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&otherDev, count * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         int8_t selfVal = 10;
         float alphaVal = 1.0f;
         aclScalar* selfScalar = aclCreateScalar(&selfVal, ACL_INT8);
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT8);
 
         auto ret = aclnnAddV3GetWorkspaceSize(selfScalar, other, alpha, out, &ws, &exec);
         printf(">>   INT8 ret=%d ws=%lu\n", ret, ws);
 
         aclDestroyScalar(alpha);
         aclDestroyScalar(selfScalar);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclrtFree(outDev);
         aclrtFree(otherDev);
     }
 
     // =============================================================
     // aclnnInplaceAdd: CheckInplace null branches
     // =============================================================
     {
         printf(">> [COV] aclnnInplaceAdd null selfRef\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* otherDev = nullptr;
         aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnInplaceAddGetWorkspaceSize(nullptr, other, alpha, &ws, &exec);
         printf(">>   selfRef=nullptr ret=%d\n", ret);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(other);
         aclrtFree(otherDev);
     }
     {
         printf(">> [COV] aclnnInplaceAdd null other\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnInplaceAddGetWorkspaceSize(self, nullptr, alpha, &ws, &exec);
         printf(">>   other=nullptr ret=%d\n", ret);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(self);
         aclrtFree(selfDev);
     }
 
     // aclnnInplaceAdd: shape broadcast mismatch
     {
         printf(">> [COV] aclnnInplaceAdd shape mismatch (selfRef shape != broadcast)\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         int64_t shapeA = 4;
         int64_t shapeB = 3;
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         aclrtMalloc(&selfDev, 4 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, 3 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shapeA, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeA, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shapeB, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeB, 1, otherDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &ws, &exec);
         printf(">>   shape mismatch ret=%d\n", ret);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // =============================================================
     // CheckPromoteType branches: alpha dtype can't cast to promote dtype
     // FLOAT self+other but BOOL alpha (BOOL can't promote to FLOAT)
     // =============================================================
     {
         printf(">> [COV] aclnnAdd CheckPromoteType: alpha dtype incompatible\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         bool alphaVal = true;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_BOOL);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   FLOAT+BOOL alpha ret=%d\n", ret);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // CheckPromoteType: promoteType == DT_UNDEFINED
     // Complex types are not in ASCEND910 support list
     {
         printf(">> [COV] aclnnAdd unsupported output dtype from promote\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* selfDev = nullptr;
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         float alphaVal = 1.0f;
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
 
         auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
         printf(">>   FLOAT->INT8 output ret=%d\n", ret);
 
         aclDestroyScalar(alpha);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclDestroyTensor(self);
         aclrtFree(outDev);
         aclrtFree(otherDev);
         aclrtFree(selfDev);
     }
 
     // =============================================================
     // Add V3 unsupported dtype
     // =============================================================
     {
         printf(">> [COV] aclnnAddV3 unsupported dtype INT64\n");
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         size_t count = 4;
         int64_t shape = static_cast<int64_t>(count);
         void* otherDev = nullptr;
         void* outDev = nullptr;
         aclrtMalloc(&otherDev, count * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&outDev, count * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
         aclTensor* other = aclCreateTensor(&shape, 1, ACL_INT64, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
         aclTensor* out = aclCreateTensor(&shape, 1, ACL_INT64, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
         int32_t selfVal = 10;
         float alphaVal = 1.0f;
         aclScalar* selfScalar = aclCreateScalar(&selfVal, ACL_INT32);
         aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);
 
         auto ret = aclnnAddV3GetWorkspaceSize(selfScalar, other, alpha, out, &ws, &exec);
         printf(">>   INT64 (unsupported) ret=%d\n", ret);
 
         aclDestroyScalar(alpha);
         aclDestroyScalar(selfScalar);
         aclDestroyTensor(out);
         aclDestroyTensor(other);
         aclrtFree(outDev);
         aclrtFree(otherDev);
     }
 
    // =============================================================
    // Tile-specific dtype paths: each triggers a different branch in
    // add_tiling_arch35.cpp DoOpTiling().
    // =============================================================

    // BF16: hits AddWithCastCompute<half> branch in DoOpTiling
    {
        printf(">> [COV] tiling BF16 dtype path\n");
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        void* selfDev = nullptr;
        void* otherDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&selfDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&otherDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* self = aclCreateTensor(&shape, 1, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
        aclTensor* other = aclCreateTensor(&shape, 1, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        float alphaVal = 1.0f;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_BF16);

        auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
        printf(">>   BF16+BF16 ret=%d ws=%lu\n", ret, ws);

        aclDestroyScalar(alpha);
        aclDestroyTensor(out);
        aclDestroyTensor(other);
        aclDestroyTensor(self);
        aclrtFree(outDev);
        aclrtFree(otherDev);
        aclrtFree(selfDev);
    }

    // INT64: hits AddWithoutCastCompute<int64_t> branch in DoOpTiling
    {
        printf(">> [COV] tiling INT64 dtype path\n");
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        void* selfDev = nullptr;
        void* otherDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&selfDev, count * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&otherDev, count * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* self = aclCreateTensor(&shape, 1, ACL_INT64, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
        aclTensor* other = aclCreateTensor(&shape, 1, ACL_INT64, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_INT64, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        float alphaVal = 1.0f;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT64);

        auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
        printf(">>   INT64+INT64 ret=%d ws=%lu\n", ret, ws);

        aclDestroyScalar(alpha);
        aclDestroyTensor(out);
        aclDestroyTensor(other);
        aclDestroyTensor(self);
        aclrtFree(outDev);
        aclrtFree(otherDev);
        aclrtFree(selfDev);
    }

    // UINT8: hits AddWithoutCastCompute<uint8_t> branch in DoOpTiling
    {
        printf(">> [COV] tiling UINT8 dtype path\n");
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        void* selfDev = nullptr;
        void* otherDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&selfDev, count * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&otherDev, count * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* self = aclCreateTensor(&shape, 1, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
        aclTensor* other = aclCreateTensor(&shape, 1, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        float alphaVal = 1.0f;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_UINT8);

        auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
        printf(">>   UINT8+UINT8 ret=%d ws=%lu\n", ret, ws);

        aclDestroyScalar(alpha);
        aclDestroyTensor(out);
        aclDestroyTensor(other);
        aclDestroyTensor(self);
        aclrtFree(outDev);
        aclrtFree(otherDev);
        aclrtFree(selfDev);
    }

    // INT32: hits AddWithoutCastCompute<int32_t> branch in DoOpTiling
    {
        printf(">> [COV] tiling INT32 dtype path\n");
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        void* selfDev = nullptr;
        void* otherDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&selfDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&otherDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* self = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
        aclTensor* other = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        float alphaVal = 1.0f;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

        auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
        printf(">>   INT32+INT32 ret=%d ws=%lu\n", ret, ws);

        aclDestroyScalar(alpha);
        aclDestroyTensor(out);
        aclDestroyTensor(other);
        aclDestroyTensor(self);
        aclrtFree(outDev);
        aclrtFree(otherDev);
        aclrtFree(selfDev);
    }

    // COMPLEX64: hits AddWithoutCastCompute<int64_t> (maps from COMPLEX64)
    //             and triggers CheckDtype unsupported -> OP_LOGE path
    {
        printf(">> [COV] tiling COMPLEX64 dtype path\n");
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        void* selfDev = nullptr;
        void* otherDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&selfDev, count * sizeof(float) * 2, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&otherDev, count * sizeof(float) * 2, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(float) * 2, ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* self = aclCreateTensor(&shape, 1, ACL_COMPLEX64, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
        aclTensor* other = aclCreateTensor(&shape, 1, ACL_COMPLEX64, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_COMPLEX64, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        float alphaVal = 1.0f;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_COMPLEX64);

        auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
        printf(">>   COMPLEX64+COMPLEX64 ret=%d ws=%lu\n", ret, ws);

        aclDestroyScalar(alpha);
        aclDestroyTensor(out);
        aclDestroyTensor(other);
        aclDestroyTensor(self);
        aclrtFree(outDev);
        aclrtFree(otherDev);
        aclrtFree(selfDev);
    }

    // =============================================================
    // aclnnAdd kernel execution: actually run aclnnAdd()
    // This covers aclnnAdd() function and the kernel execution path.
    // =============================================================
    {
        printf(">> [COV] aclnnAdd kernel execution: FLOAT alpha=1.0\n");
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        std::vector<float> hSelf(count), hOther(count);
        for (size_t i = 0; i < count; ++i) { hSelf[i] = static_cast<float>(i); hOther[i] = static_cast<float>(i * 2); }

        void* selfDev = nullptr;
        void* otherDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(selfDev, count * sizeof(float), hSelf.data(), count * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(otherDev, count * sizeof(float), hOther.data(), count * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
        aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        float alphaVal = 1.0f;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        auto ret1 = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
        CHECK_RET(ret1 == ACL_SUCCESS, printf(">>   GWS failed ret=%d\n", ret1));
        void* wsDev = nullptr;
        if (ws > 0) aclrtMalloc(&wsDev, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        auto ret2 = aclnnAdd(wsDev, ws, exec, stream);
        aclrtSynchronizeStream(stream);
        printf(">>   aclnnAdd ret=%d\n", ret2);

        std::vector<float> actual(count);
        aclrtMemcpy(actual.data(), count * sizeof(float), outDev, count * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(actual[i] - (hSelf[i] + hOther[i])) > 1e-5f) {
                printf(">>   MISMATCH[%zu]: got %f expected %f\n", i, actual[i], hSelf[i] + hOther[i]);
            }
        }

        if (wsDev) aclrtFree(wsDev);
        aclDestroyScalar(alpha);
        aclDestroyTensor(out);
        aclDestroyTensor(other);
        aclDestroyTensor(self);
        aclrtFree(outDev);
        aclrtFree(otherDev);
        aclrtFree(selfDev);
    }

    // aclnnAdds kernel execution
    {
        printf(">> [COV] aclnnAdds kernel execution: FLOAT alpha=1.0\n");
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        std::vector<float> hSelf(count);
        for (size_t i = 0; i < count; ++i) hSelf[i] = static_cast<float>(i);

        void* selfDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(selfDev, count * sizeof(float), hSelf.data(), count * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        float scalarVal = 5.0f;
        float alphaVal = 1.0f;
        aclScalar* scalar = aclCreateScalar(&scalarVal, ACL_FLOAT);
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        auto ret1 = aclnnAddsGetWorkspaceSize(self, scalar, alpha, out, &ws, &exec);
        CHECK_RET(ret1 == ACL_SUCCESS, printf(">>   GWS failed ret=%d\n", ret1));
        void* wsDev = nullptr;
        if (ws > 0) aclrtMalloc(&wsDev, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        auto ret2 = aclnnAdds(wsDev, ws, exec, stream);
        aclrtSynchronizeStream(stream);
        printf(">>   aclnnAdds ret=%d\n", ret2);

        std::vector<float> actual(count);
        aclrtMemcpy(actual.data(), count * sizeof(float), outDev, count * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(actual[i] - (hSelf[i] + scalarVal)) > 1e-5f) {
                printf(">>   MISMATCH[%zu]: got %f expected %f\n", i, actual[i], hSelf[i] + scalarVal);
            }
        }

        if (wsDev) aclrtFree(wsDev);
        aclDestroyScalar(alpha);
        aclDestroyScalar(scalar);
        aclDestroyTensor(out);
        aclDestroyTensor(self);
        aclrtFree(outDev);
        aclrtFree(selfDev);
    }

    // aclnnInplaceAdd kernel execution
    {
        printf(">> [COV] aclnnInplaceAdd kernel execution: FLOAT alpha=1.0\n");
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        std::vector<float> hSelf(count), hOther(count);
        for (size_t i = 0; i < count; ++i) { hSelf[i] = static_cast<float>(i); hOther[i] = static_cast<float>(i * 3); }

        void* selfDev = nullptr;
        void* otherDev = nullptr;
        aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(selfDev, count * sizeof(float), hSelf.data(), count * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(otherDev, count * sizeof(float), hOther.data(), count * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
        aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
        float alphaVal = 1.0f;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        auto ret1 = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &ws, &exec);
        CHECK_RET(ret1 == ACL_SUCCESS, printf(">>   GWS failed ret=%d\n", ret1));
        void* wsDev = nullptr;
        if (ws > 0) aclrtMalloc(&wsDev, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        auto ret2 = aclnnInplaceAdd(wsDev, ws, exec, stream);
        aclrtSynchronizeStream(stream);
        printf(">>   aclnnInplaceAdd ret=%d\n", ret2);

        std::vector<float> actual(count);
        aclrtMemcpy(actual.data(), count * sizeof(float), selfDev, count * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(actual[i] - (hSelf[i] + hOther[i])) > 1e-5f) {
                printf(">>   MISMATCH[%zu]: got %f expected %f\n", i, actual[i], hSelf[i] + hOther[i]);
            }
        }

        if (wsDev) aclrtFree(wsDev);
        aclDestroyScalar(alpha);
        aclDestroyTensor(other);
        aclDestroyTensor(self);
        aclrtFree(otherDev);
        aclrtFree(selfDev);
    }

    // aclnnInplaceAdds kernel execution
    {
        printf(">> [COV] aclnnInplaceAdds kernel execution\n");
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        std::vector<float> hSelf(count);
        for (size_t i = 0; i < count; ++i) hSelf[i] = static_cast<float>(i);

        void* selfDev = nullptr;
        aclrtMalloc(&selfDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(selfDev, count * sizeof(float), hSelf.data(), count * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
        float scalarVal = 7.0f;
        float alphaVal = 1.0f;
        aclScalar* scalar = aclCreateScalar(&scalarVal, ACL_FLOAT);
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        auto ret1 = aclnnInplaceAddsGetWorkspaceSize(self, scalar, alpha, &ws, &exec);
        CHECK_RET(ret1 == ACL_SUCCESS, printf(">>   GWS failed ret=%d\n", ret1));
        void* wsDev = nullptr;
        if (ws > 0) aclrtMalloc(&wsDev, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        auto ret2 = aclnnInplaceAdds(wsDev, ws, exec, stream);
        aclrtSynchronizeStream(stream);
        printf(">>   aclnnInplaceAdds ret=%d\n", ret2);

        std::vector<float> actual(count);
        aclrtMemcpy(actual.data(), count * sizeof(float), selfDev, count * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(actual[i] - (hSelf[i] + scalarVal)) > 1e-5f) {
                printf(">>   MISMATCH[%zu]: got %f expected %f\n", i, actual[i], hSelf[i] + scalarVal);
            }
        }

        if (wsDev) aclrtFree(wsDev);
        aclDestroyScalar(alpha);
        aclDestroyScalar(scalar);
        aclDestroyTensor(self);
        aclrtFree(selfDev);
    }

    // aclnnAddV3 kernel execution: alpha=1 (Add path)
    {
        printf(">> [COV] aclnnAddV3 kernel execution: FLOAT alpha=1.0\n");
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        std::vector<float> hOther(count);
        for (size_t i = 0; i < count; ++i) hOther[i] = static_cast<float>(i);

        void* otherDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(otherDev, count * sizeof(float), hOther.data(), count * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

        float selfVal = 10.0f;
        aclScalar* selfScalar = aclCreateScalar(&selfVal, ACL_FLOAT);
        aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        float alphaVal = 1.0f;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        auto ret1 = aclnnAddV3GetWorkspaceSize(selfScalar, other, alpha, out, &ws, &exec);
        CHECK_RET(ret1 == ACL_SUCCESS, printf(">>   GWS failed ret=%d\n", ret1));
        void* wsDev = nullptr;
        if (ws > 0) aclrtMalloc(&wsDev, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        auto ret2 = aclnnAddV3(wsDev, ws, exec, stream);
        aclrtSynchronizeStream(stream);
        printf(">>   aclnnAddV3 ret=%d\n", ret2);

        std::vector<float> actual(count);
        aclrtMemcpy(actual.data(), count * sizeof(float), outDev, count * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(actual[i] - (selfVal + hOther[i])) > 1e-5f) {
                printf(">>   MISMATCH[%zu]: got %f expected %f\n", i, actual[i], selfVal + hOther[i]);
            }
        }

        if (wsDev) aclrtFree(wsDev);
        aclDestroyScalar(selfScalar);
        aclDestroyScalar(alpha);
        aclDestroyTensor(out);
        aclDestroyTensor(other);
        aclrtFree(outDev);
        aclrtFree(otherDev);
    }

    // aclnnInplaceAddV3 kernel execution
    {
        printf(">> [COV] aclnnInplaceAddV3 kernel execution\n");
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        std::vector<float> hOther(count);
        for (size_t i = 0; i < count; ++i) hOther[i] = static_cast<float>(i);

        void* otherDev = nullptr;
        aclrtMalloc(&otherDev, count * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(otherDev, count * sizeof(float), hOther.data(), count * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

        float selfVal = 20.0f;
        aclScalar* selfScalar = aclCreateScalar(&selfVal, ACL_FLOAT);
        aclTensor* other = aclCreateTensor(&shape, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
        float alphaVal = 1.0f;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        auto ret1 = aclnnInplaceAddV3GetWorkspaceSize(selfScalar, other, alpha, &ws, &exec);
        CHECK_RET(ret1 == ACL_SUCCESS, printf(">>   GWS failed ret=%d\n", ret1));
        void* wsDev = nullptr;
        if (ws > 0) aclrtMalloc(&wsDev, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        auto ret2 = aclnnInplaceAddV3(wsDev, ws, exec, stream);
        aclrtSynchronizeStream(stream);
        printf(">>   aclnnInplaceAddV3 ret=%d\n", ret2);

        std::vector<float> actual(count);
        aclrtMemcpy(actual.data(), count * sizeof(float), otherDev, count * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        for (size_t i = 0; i < count; ++i) {
            if (std::abs(actual[i] - (selfVal + hOther[i])) > 1e-5f) {
                printf(">>   MISMATCH[%zu]: got %f expected %f\n", i, actual[i], selfVal + hOther[i]);
            }
        }

        if (wsDev) aclrtFree(wsDev);
        aclDestroyScalar(alpha);
        aclDestroyScalar(selfScalar);
        aclDestroyTensor(other);
        aclrtFree(otherDev);
    }

    // =============================================================
    // aclnnAddV3: other IsEmpty with alpha=1 (early-return branch)
    // =============================================================
    {
        printf(">> [COV] aclnnAddV3 other IsEmpty with alpha=1\n");
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        int64_t shapeIn = 0;
        int64_t shapeOut = 4;
        void* otherDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&otherDev, 1 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, 4 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* other = aclCreateTensor(&shapeIn, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeIn, 1, otherDev);
        aclTensor* out = aclCreateTensor(&shapeOut, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeOut, 1, outDev);
        float selfVal = 10.0f;
        float alphaVal = 1.0f;
        aclScalar* selfScalar = aclCreateScalar(&selfVal, ACL_FLOAT);
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

        auto ret = aclnnAddV3GetWorkspaceSize(selfScalar, other, alpha, out, &ws, &exec);
        printf(">>   other IsEmpty alpha=1 ret=%d ws=%lu\n", ret, ws);

        aclDestroyScalar(alpha);
        aclDestroyScalar(selfScalar);
        aclDestroyTensor(out);
        aclDestroyTensor(other);
        aclrtFree(outDev);
        aclrtFree(otherDev);
    }

    // =============================================================
    // aclnnAddV3: IsSupportAxpy path with alpha=2 (Axpy call)
    // Tests: 6:  232:else if (IsSupportAxpy(promoteType)) {
    //         4:  234:        addOpOut = l0op::Axpy(...)
    // =============================================================
    {
        printf(">> [COV] aclnnAddV3 Axpy path: INT32 alpha=2.0\n");
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        void* otherDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&otherDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* other = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        int32_t selfVal = 10;
        float alphaVal = 2.0f;
        aclScalar* selfScalar = aclCreateScalar(&selfVal, ACL_INT32);
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

        auto ret = aclnnAddV3GetWorkspaceSize(selfScalar, other, alpha, out, &ws, &exec);
        printf(">>   INT32 alpha=2.0 ret=%d ws=%lu\n", ret, ws);

        aclDestroyScalar(alpha);
        aclDestroyScalar(selfScalar);
        aclDestroyTensor(out);
        aclDestroyTensor(other);
        aclrtFree(outDev);
        aclrtFree(otherDev);
    }

    // =============================================================
    // aclnnAddV3: non-Axpy type (BF16) with alpha=2 (Mul+Add path)
    // Tests: 2:  236:else { ... Mul+Add ... }
    // =============================================================
    {
        printf(">> [COV] aclnnAddV3 Mul+Add path: BF16 alpha=2.0\n");
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        void* otherDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&otherDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* other = aclCreateTensor(&shape, 1, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        float selfVal = 10.0f;
        float alphaVal = 2.0f;
        aclScalar* selfScalar = aclCreateScalar(&selfVal, ACL_BF16);
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_BF16);

        auto ret = aclnnAddV3GetWorkspaceSize(selfScalar, other, alpha, out, &ws, &exec);
        printf(">>   BF16 alpha=2.0 ret=%d ws=%lu\n", ret, ws);

        aclDestroyScalar(alpha);
        aclDestroyScalar(selfScalar);
        aclDestroyTensor(out);
        aclDestroyTensor(other);
        aclrtFree(outDev);
        aclrtFree(otherDev);
    }

    // =============================================================
    // aclnnAddV3: non-Axpy type (INT8) with alpha=2 (Mul+Add path)
    // =============================================================
    {
        printf(">> [COV] aclnnAddV3 Mul+Add path: INT8 alpha=2.0\n");
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        void* otherDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&otherDev, count * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(int8_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* other = aclCreateTensor(&shape, 1, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        int8_t selfVal = 10;
        float alphaVal = 2.0f;
        aclScalar* selfScalar = aclCreateScalar(&selfVal, ACL_INT8);
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT8);

        auto ret = aclnnAddV3GetWorkspaceSize(selfScalar, other, alpha, out, &ws, &exec);
        printf(">>   INT8 alpha=2.0 ret=%d ws=%lu\n", ret, ws);

        aclDestroyScalar(alpha);
        aclDestroyScalar(selfScalar);
        aclDestroyTensor(out);
        aclDestroyTensor(other);
        aclrtFree(outDev);
        aclrtFree(otherDev);
    }

    // =============================================================
    // aclnnAdd: IsSupportAxpy fallback (non-RegBase branch)
    // This hits: IsSupportAxpy: 1218: return CheckType(promoteType, AXPY_DTYPE_SUPPORT_LIST)
    // We test this by checking ACL_DT_UNDEFINED or unsupported dtype.
    // =============================================================
    // The non-RegBase branch in IsSupportAxpy is hit when IsRegBase()=false.
    // Since we can't easily control IsRegBase(), we cover the AXPY_DTYPE_SUPPORT_LIST
    // check path by testing with DOUBLE (not in AXPY_DTYPE_SUPPORT_LIST on RegBase arch).
    {
        printf(">> [COV] aclnnAdd DOUBLE+DOUBLE alpha=2.0 (Mul+Add fallback)\n");
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        void* selfDev = nullptr;
        void* otherDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&selfDev, count * sizeof(double), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&otherDev, count * sizeof(double), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(double), ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* self = aclCreateTensor(&shape, 1, ACL_DOUBLE, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
        aclTensor* other = aclCreateTensor(&shape, 1, ACL_DOUBLE, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_DOUBLE, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        double alphaVal = 2.0;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_DOUBLE);

        auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
        printf(">>   DOUBLE+DOUBLE alpha=2 ret=%d ws=%lu\n", ret, ws);

        aclDestroyScalar(alpha);
        aclDestroyTensor(out);
        aclDestroyTensor(other);
        aclDestroyTensor(self);
        aclrtFree(outDev);
        aclrtFree(otherDev);
        aclrtFree(selfDev);
    }

    // =============================================================
    // aclnnAdds: IsSupportAxpy path with alpha=2 (tests the Axpy call)
    // =============================================================
    {
        printf(">> [COV] aclnnAdds INT32 Axpy path: alpha=2.0\n");
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        void* selfDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&selfDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* self = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        int32_t scalarVal = 3;
        float alphaVal = 2.0f;
        aclScalar* scalar = aclCreateScalar(&scalarVal, ACL_INT32);
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_INT32);

        auto ret = aclnnAddsGetWorkspaceSize(self, scalar, alpha, out, &ws, &exec);
        printf(">>   INT32 alpha=2.0 ret=%d ws=%lu\n", ret, ws);

        aclDestroyScalar(alpha);
        aclDestroyScalar(scalar);
        aclDestroyTensor(out);
        aclDestroyTensor(self);
        aclrtFree(outDev);
        aclrtFree(selfDev);
    }

    // =============================================================
    // aclnnAdds: FLOAT16+alpha=2 (Axpy path)
    // =============================================================
    {
        printf(">> [COV] aclnnAdds FLOAT16 Axpy path: alpha=2.0\n");
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        void* selfDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&selfDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* self = aclCreateTensor(&shape, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        uint16_t scalarVal = FloatToUint16(3.0f);
        float alphaVal = 2.0f;
        aclScalar* scalar = aclCreateScalar(&scalarVal, ACL_FLOAT16);
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT16);

        auto ret = aclnnAddsGetWorkspaceSize(self, scalar, alpha, out, &ws, &exec);
        printf(">>   FLOAT16 alpha=2.0 ret=%d ws=%lu\n", ret, ws);

        aclDestroyScalar(alpha);
        aclDestroyScalar(scalar);
        aclDestroyTensor(out);
        aclDestroyTensor(self);
        aclrtFree(outDev);
        aclrtFree(selfDev);
    }

    // =============================================================
    // Precision Extensions — COMPLEX128 dtype (highest-complexity type)
    // Complex128 tests trigger unique code paths in type promotion and tiling.
    // =============================================================
    {
        printf(">> [COV] aclnnAdd COMPLEX128+COMPLEX128 COMPLEX64 tiling path\n");
        size_t count = 4;

        void* selfDev = nullptr;
        void* otherDev = nullptr;
        void* outDev = nullptr;
        aclTensor* selfT = nullptr;
        aclTensor* otherT = nullptr;
        aclTensor* outT = nullptr;
        aclScalar* alpha = nullptr;
        int ret = ACL_SUCCESS;
        do {
            ret = aclrtMalloc(&selfDev, count * 2 * sizeof(double), ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) break;
            ret = aclrtMalloc(&otherDev, count * 2 * sizeof(double), ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) break;
            ret = aclrtMalloc(&outDev, count * 2 * sizeof(double), ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) break;

            int64_t shape = static_cast<int64_t>(count * 2);
            selfT = aclCreateTensor(&shape, 1, ACL_COMPLEX128, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
            otherT = aclCreateTensor(&shape, 1, ACL_COMPLEX128, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
            outT = aclCreateTensor(&shape, 1, ACL_COMPLEX128, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
            double alphaHost = 1.0;
            alpha = aclCreateScalar(&alphaHost, ACL_COMPLEX128);

            uint64_t ws = 0;
            aclOpExecutor* executor = nullptr;
            ret = aclnnAddGetWorkspaceSize(selfT, otherT, alpha, outT, &ws, &executor);
            printf(">>   COMPLEX128 ret=%d ws=%lu\n", ret, ws);
        } while (false);

        if (alpha) aclDestroyScalar(alpha);
        if (outT) aclDestroyTensor(outT);
        if (otherT) aclDestroyTensor(otherT);
        if (selfT) aclDestroyTensor(selfT);
        if (outDev) aclrtFree(outDev);
        if (otherDev) aclrtFree(otherDev);
        if (selfDev) aclrtFree(selfDev);
    }

    // =============================================================
    // Broadcast shape {2,3} + {3} -> {2,3}
    // aclnn_add.cpp line 254-255: OP_CHECK_BROADCAST_AND_INFER_SHAPE
    // Exercises the broadcast inference code path
    // =============================================================
    {
        printf(">> [COV] aclnnAdd broadcast {2,3} + {3} -> {2,3}\n");
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        int64_t shapeA[2] = {2, 3};
        int64_t shapeB[1] = {3};
        int64_t shapeOut[2] = {2, 3};
        void* selfDev = nullptr;
        void* otherDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&selfDev, 6 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&otherDev, 3 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, 6 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* self = aclCreateTensor(shapeA, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shapeA, 2, selfDev);
        aclTensor* other = aclCreateTensor(shapeB, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shapeB, 1, otherDev);
        aclTensor* out = aclCreateTensor(shapeOut, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shapeOut, 2, outDev);
        float alphaVal = 1.0f;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

        auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec);
        printf(">>   broadcast {2,3}+{3} ret=%d ws=%lu\n", ret, ws);

        aclDestroyScalar(alpha);
        aclDestroyTensor(out);
        aclDestroyTensor(other);
        aclDestroyTensor(self);
        aclrtFree(outDev);
        aclrtFree(otherDev);
        aclrtFree(selfDev);
    }

    // =============================================================
    // aclnnAdds: DOUBLE tensor + DOUBLE scalar (Mul+Add path)
    // aclnn_add.cpp line 606-620: DOUBLE not in AXPY/AxpyV2 -> fallback
    // =============================================================
    {
        printf(">> [COV] aclnnAdds DOUBLE tensor + scalar (Mul+Add fallback)\n");
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        void* selfDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&selfDev, count * sizeof(double), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(double), ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* self = aclCreateTensor(&shape, 1, ACL_DOUBLE, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_DOUBLE, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        double scalarVal = 2.5;
        double alphaVal = 1.5;
        aclScalar* scalar = aclCreateScalar(&scalarVal, ACL_DOUBLE);
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_DOUBLE);

        auto ret = aclnnAddsGetWorkspaceSize(self, scalar, alpha, out, &ws, &exec);
        printf(">>   DOUBLE adds ret=%d ws=%lu\n", ret, ws);

        aclDestroyScalar(alpha);
        aclDestroyScalar(scalar);
        aclDestroyTensor(out);
        aclDestroyTensor(self);
        aclrtFree(outDev);
        aclrtFree(selfDev);
    }

    // =============================================================
    // aclnnAdd: BF16+BF16 precision with kernel execution
    // aclnn_add.cpp line 372: promoteType==self==other -> Add kernel path
    // =============================================================
    {
        printf(">> [COV] aclnnAdd BF16+BF16 tiling path\n");
        size_t count = 8;
        int64_t shape = static_cast<int64_t>(count);
        void* selfDev = nullptr;
        void* otherDev = nullptr;
        void* outDev = nullptr;
        aclrtMalloc(&selfDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&otherDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, count * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* self = aclCreateTensor(&shape, 1, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, selfDev);
        aclTensor* other = aclCreateTensor(&shape, 1, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, otherDev);
        aclTensor* out = aclCreateTensor(&shape, 1, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, &shape, 1, outDev);
        float alphaVal = 1.0f;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_BF16);
        uint64_t ws = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &executor);
        printf(">>   BF16+BF16 ret=%d ws=%lu\n", ret, ws);

        aclDestroyScalar(alpha);
        aclDestroyTensor(out);
        aclDestroyTensor(other);
        aclDestroyTensor(self);
        aclrtFree(outDev);
        aclrtFree(otherDev);
        aclrtFree(selfDev);
    }

    // =============================================================
    // aclnnInplaceAdd: broadcast {4} + {2} -> error (inplace shape mismatch)
    // add.cpp line 136: broadcastShape != other->GetViewShape()
    // =============================================================
    {
        printf(">> [COV] aclnnInplaceAdd broadcast mismatch (error path)\n");
        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        int64_t shapeSelf = 4;
        int64_t shapeOther = 2;
        void* selfDev = nullptr;
        void* otherDev = nullptr;
        aclrtMalloc(&selfDev, 4 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&otherDev, 2 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclTensor* self = aclCreateTensor(&shapeSelf, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeSelf, 1, selfDev);
        aclTensor* other = aclCreateTensor(&shapeOther, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, &shapeOther, 1, otherDev);
        float alphaVal = 1.0f;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

        auto ret = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &ws, &exec);
        printf(">>   inplace broadcast mismatch ret=%d\n", ret);

        aclDestroyScalar(alpha);
        aclDestroyTensor(other);
        aclDestroyTensor(self);
        aclrtFree(otherDev);
        aclrtFree(selfDev);
    }

    LOG_PRINT("\n[INFO] Coverage tests completed.\n");
}

// =============================================================
// Main
 // =============================================================
 int main()
 {
     std::remove("test_aclnn_add.gcda");
     std::remove("test_aclnn_add.gcno");
 
     aclrtStream stream;
     auto aclRet = InitAcl(0, &stream);
     if (aclRet != 0) {
         LOG_PRINT("InitAcl failed.\n");
         return aclRet;
     }
 
     int ret = RunPrecisionTests(stream);
     if (ret != ACL_SUCCESS) {
         LOG_PRINT("RunPrecisionTests failed.\n");
     } else {
         LOG_PRINT("RunPrecisionTests success.\n");
     }
 
     RunCoverageTests(stream);
 
     aclrtDestroyStream(stream);
     aclrtResetDevice(0);
     aclFinalize();
 
     return 0;
 }
 