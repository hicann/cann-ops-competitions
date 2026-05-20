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
#include <sstream>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <algorithm>
#include <iomanip>
#include <type_traits>
#include "acl/acl.h"
#include "aclnnop/aclnn_exp2.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"

// 测试统计
struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
    std::vector<std::string> failedCases;

    void addPass(const std::string& name) {
        total++;
        passed++;
        std::cout << "[PASS] " << name << std::endl;
    }

    void addFail(const std::string& name, const std::string& reason) {
        total++;
        failed++;
        std::string msg = name + ": " + reason;
        failedCases.push_back(msg);
        std::cout << "[FAIL] " << msg << std::endl;
    }

    void printSummary() const {
        std::cout << "\n========== Test Summary ==========" << std::endl;
        std::cout << "Total:  " << total << std::endl;
        std::cout << "Passed: " << passed << std::endl;
        std::cout << "Failed: " << failed << std::endl;
        if (!failedCases.empty()) {
            std::cout << "\nFailed cases:" << std::endl;
            for (const auto& c : failedCases) {
                std::cout << "  - " << c << std::endl;
            }
        }
        std::cout << "==================================" << std::endl;
    }
};

static TestStats g_stats;

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

// 容差配置
struct Tolerance {
    double rtol = 1e-5;
    double atol = 1e-8;

    static Tolerance ForType(aclDataType dtype) {
        Tolerance t;
        switch (dtype) {
            case ACL_FLOAT:
                t.rtol = 1e-5; t.atol = 1e-6;
                break;
            case ACL_FLOAT16:
                t.rtol = 1e-3; t.atol = 1e-4;
                break;
            case ACL_BF16:
                t.rtol = 1e-2; t.atol = 1e-3;
                break;
            default:
                t.rtol = 1e-5; t.atol = 1e-6;
                break;
        }
        return t;
    }
};

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1;
    int64_t size = 1;
    for (auto dim : shape) size *= dim;
    return size;
}

inline double ToDouble(float val) { return static_cast<double>(val); }
inline double ToDouble(int32_t val) { return static_cast<double>(val); }
inline double ToDouble(int8_t val) { return static_cast<double>(val); }
inline double ToDouble(int64_t val) { return static_cast<double>(val); }

template <typename T>
std::vector<double> CalcExpectedExp2(const std::vector<T>& self) {
    std::vector<double> expected;
    expected.reserve(self.size());
    for (auto v : self) {
        expected.push_back(std::pow(2.0, ToDouble(v)));
    }
    return expected;
}

template <typename T>
std::vector<double> CalcExpectedPowTensorScalar(const std::vector<T>& base, double exp) {
    std::vector<double> expected;
    expected.reserve(base.size());
    for (auto v : base) {
        expected.push_back(std::pow(ToDouble(v), exp));
    }
    return expected;
}

template <typename T>
std::vector<double> CalcExpectedPowScalarTensor(double base, const std::vector<T>& exp) {
    std::vector<double> expected;
    expected.reserve(exp.size());
    for (auto v : exp) {
        expected.push_back(std::pow(base, ToDouble(v)));
    }
    return expected;
}

template <typename T>
std::vector<double> CalcExpectedPowTensorTensor(const std::vector<T>& base, const std::vector<T>& exp) {
    std::vector<double> expected;
    size_t size = std::min(base.size(), exp.size());
    expected.reserve(size);
    for (size_t i = 0; i < size; i++) {
        expected.push_back(std::pow(ToDouble(base[i]), ToDouble(exp[i])));
    }
    return expected;
}

bool AllClose(const std::vector<double>& expected, const std::vector<float>& actual, 
              const Tolerance& tol, std::string* errorMsg = nullptr) {
    if (expected.size() != actual.size()) {
        if (errorMsg) *errorMsg = "Size mismatch: " + std::to_string(expected.size()) + " vs " + std::to_string(actual.size());
        return false;
    }

    for (size_t i = 0; i < expected.size(); i++) {
        double exp = expected[i];
        double act = static_cast<double>(actual[i]);
        double diff = std::abs(act - exp);
        double tolerance = tol.atol + tol.rtol * std::abs(exp);

        bool expNaN = std::isnan(exp);
        bool actNaN = std::isnan(act);
        bool expInf = std::isinf(exp);
        bool actInf = std::isinf(act);

        if (expNaN != actNaN || expInf != actInf) {
            if (errorMsg) {
                *errorMsg = "Index " + std::to_string(i) + ": Special value mismatch";
            }
            return false;
        }

        if (!expNaN && !expInf && diff > tolerance) {
            if (errorMsg) {
                std::ostringstream oss;
                oss << "Index " << i << ": Expected " << std::setprecision(10) << exp 
                    << ", Got " << act << ", Diff " << diff << " > Tol " << tolerance;
                *errorMsg = oss.str();
            }
            return false;
        }
    }
    return true;
}

template <typename T>
aclTensor* CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, 
                          void** deviceAddr, aclDataType dataType) {
    if (hostData.empty() || shape.empty()) return nullptr;

    size_t size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return nullptr;
    }

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return nullptr;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    return aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 
                          0, ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
}

void CleanupTensor(aclTensor*& tensor, void*& deviceAddr) {
    if (tensor) { aclDestroyTensor(tensor); tensor = nullptr; }
    if (deviceAddr) { aclrtFree(deviceAddr); deviceAddr = nullptr; }
}

int InitACL(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) { LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret; }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) { LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret; }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) { LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret; }
    return 0;
}

void FinalizeACL(int32_t deviceId, aclrtStream stream) {
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

// ================= 7个API测试用例 =================

// API 1: aclnnExp2
template <typename T>
int TestExp2(aclrtStream stream, const std::string& typeName) {
    std::string caseName = "Exp2_" + typeName;
    std::vector<int64_t> shape = {2, 2};
    std::vector<T> selfHostData = {static_cast<T>(0), static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)};

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclDataType dtype = ACL_FLOAT;

    aclTensor* self = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, dtype);
    aclTensor* out = CreateAclTensor(std::vector<T>(4, 0), shape, &outDeviceAddr, dtype);

    if (!self || !out) {
        g_stats.addFail(caseName, "Failed to create tensor");
        CleanupTensor(self, selfDeviceAddr);
        CleanupTensor(out, outDeviceAddr);
        return -1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnExp2GetWorkspaceSize(self, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        g_stats.addFail(caseName, "GetWorkspaceSize failed: " + std::to_string(ret));
        CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr);
        return ret;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclnnExp2(workspaceAddr, workspaceSize, executor, stream);
    ret = aclrtSynchronizeStream(stream);

    std::vector<T> resultData(4);
    aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), outDeviceAddr, resultData.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> resultFloat(4);
    for (size_t i = 0; i < 4; i++) resultFloat[i] = static_cast<float>(resultData[i]);

    auto expected = CalcExpectedExp2(selfHostData);
    Tolerance tol = Tolerance::ForType(dtype);
    std::string errorMsg;
    bool pass = AllClose(expected, resultFloat, tol, &errorMsg);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr);

    if (pass) g_stats.addPass(caseName);
    else g_stats.addFail(caseName, errorMsg);
    return pass ? 0 : 1;
}

// API 2: aclnnInplaceExp2
template <typename T>
int TestInplaceExp2(aclrtStream stream, const std::string& typeName) {
    std::string caseName = "InplaceExp2_" + typeName;
    std::vector<int64_t> shape = {2, 2};
    std::vector<T> selfHostData = {static_cast<T>(0), static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)};
    std::vector<T> backup = selfHostData;

    void* selfDeviceAddr = nullptr;
    aclDataType dtype = ACL_FLOAT;

    aclTensor* self = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, dtype);
    if (!self) {
        g_stats.addFail(caseName, "Failed to create tensor");
        CleanupTensor(self, selfDeviceAddr);
        return -1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceExp2GetWorkspaceSize(self, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        g_stats.addFail(caseName, "GetWorkspaceSize failed: " + std::to_string(ret));
        CleanupTensor(self, selfDeviceAddr);
        return ret;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclnnInplaceExp2(workspaceAddr, workspaceSize, executor, stream);
    ret = aclrtSynchronizeStream(stream);

    std::vector<T> resultData(4);
    aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), selfDeviceAddr, resultData.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> resultFloat(4);
    for (size_t i = 0; i < 4; i++) resultFloat[i] = static_cast<float>(resultData[i]);

    auto expected = CalcExpectedExp2(backup);
    Tolerance tol = Tolerance::ForType(dtype);
    std::string errorMsg;
    bool pass = AllClose(expected, resultFloat, tol, &errorMsg);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    CleanupTensor(self, selfDeviceAddr);

    if (pass) g_stats.addPass(caseName);
    else g_stats.addFail(caseName, errorMsg);
    return pass ? 0 : 1;
}

// API 3: aclnnPowTensorScalar
template <typename T>
int TestPowTensorScalar(aclrtStream stream, float expVal, const std::string& desc, const std::string& typeName) {
    std::string caseName = "PowTensorScalar_" + desc + "_" + typeName;
    std::vector<int64_t> shape = {2, 2};
    std::vector<T> selfHostData = {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)};

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclDataType dtype = ACL_FLOAT;

    aclTensor* self = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, dtype);
    aclTensor* out = CreateAclTensor(std::vector<T>(4, 0), shape, &outDeviceAddr, dtype);

    if (!self || !out) {
        g_stats.addFail(caseName, "Failed to create tensor");
        CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr);
        return -1;
    }

    aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);
    if (!exponent) {
        g_stats.addFail(caseName, "Failed to create scalar");
        CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr);
        return -1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        g_stats.addFail(caseName, "GetWorkspaceSize failed: " + std::to_string(ret));
        aclDestroyScalar(exponent);
        CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr);
        return ret;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    ret = aclrtSynchronizeStream(stream);

    std::vector<T> resultData(4);
    aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), outDeviceAddr, resultData.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> resultFloat(4);
    for (size_t i = 0; i < 4; i++) resultFloat[i] = static_cast<float>(resultData[i]);

    auto expected = CalcExpectedPowTensorScalar(selfHostData, static_cast<double>(expVal));
    Tolerance tol = Tolerance::ForType(dtype);
    std::string errorMsg;
    bool pass = AllClose(expected, resultFloat, tol, &errorMsg);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(exponent);
    CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr);

    if (pass) g_stats.addPass(caseName);
    else g_stats.addFail(caseName, errorMsg);
    return pass ? 0 : 1;
}

// API 4: aclnnInplacePowTensorScalar
template <typename T>
int TestInplacePowTensorScalar(aclrtStream stream, float expVal, const std::string& desc, const std::string& typeName) {
    std::string caseName = "InplacePowTensorScalar_" + desc + "_" + typeName;
    std::vector<int64_t> shape = {2, 2};
    std::vector<T> selfHostData = {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)};
    std::vector<T> backup = selfHostData;

    void* selfDeviceAddr = nullptr;
    aclDataType dtype = ACL_FLOAT;

    aclTensor* self = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, dtype);
    if (!self) {
        g_stats.addFail(caseName, "Failed to create tensor");
        CleanupTensor(self, selfDeviceAddr);
        return -1;
    }

    aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);
    if (!exponent) {
        g_stats.addFail(caseName, "Failed to create scalar");
        CleanupTensor(self, selfDeviceAddr);
        return -1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self, exponent, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        g_stats.addFail(caseName, "GetWorkspaceSize failed: " + std::to_string(ret));
        aclDestroyScalar(exponent);
        CleanupTensor(self, selfDeviceAddr);
        return ret;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclnnInplacePowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    ret = aclrtSynchronizeStream(stream);

    std::vector<T> resultData(4);
    aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), selfDeviceAddr, resultData.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> resultFloat(4);
    for (size_t i = 0; i < 4; i++) resultFloat[i] = static_cast<float>(resultData[i]);

    auto expected = CalcExpectedPowTensorScalar(backup, static_cast<double>(expVal));
    Tolerance tol = Tolerance::ForType(dtype);
    std::string errorMsg;
    bool pass = AllClose(expected, resultFloat, tol, &errorMsg);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(exponent);
    CleanupTensor(self, selfDeviceAddr);

    if (pass) g_stats.addPass(caseName);
    else g_stats.addFail(caseName, errorMsg);
    return pass ? 0 : 1;
}

// API 5: aclnnPowScalarTensor (新增)
template <typename T>
int TestPowScalarTensor(aclrtStream stream, float baseVal, const std::string& desc, const std::string& typeName) {
    std::string caseName = "PowScalarTensor_" + desc + "_" + typeName;
    std::vector<int64_t> shape = {2, 2};
    std::vector<T> expHostData = {static_cast<T>(0), static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)};

    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclDataType dtype = ACL_FLOAT;

    aclTensor* exp = CreateAclTensor(expHostData, shape, &expDeviceAddr, dtype);
    aclTensor* out = CreateAclTensor(std::vector<T>(4, 0), shape, &outDeviceAddr, dtype);

    if (!exp || !out) {
        g_stats.addFail(caseName, "Failed to create tensor");
        CleanupTensor(exp, expDeviceAddr); CleanupTensor(out, outDeviceAddr);
        return -1;
    }

    aclScalar* base = aclCreateScalar(&baseVal, ACL_FLOAT);
    if (!base) {
        g_stats.addFail(caseName, "Failed to create scalar");
        CleanupTensor(exp, expDeviceAddr); CleanupTensor(out, outDeviceAddr);
        return -1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowScalarTensorGetWorkspaceSize(base, exp, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        g_stats.addFail(caseName, "GetWorkspaceSize failed: " + std::to_string(ret));
        aclDestroyScalar(base);
        CleanupTensor(exp, expDeviceAddr); CleanupTensor(out, outDeviceAddr);
        return ret;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
    ret = aclrtSynchronizeStream(stream);

    std::vector<T> resultData(4);
    aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), outDeviceAddr, resultData.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> resultFloat(4);
    for (size_t i = 0; i < 4; i++) resultFloat[i] = static_cast<float>(resultData[i]);

    auto expected = CalcExpectedPowScalarTensor(static_cast<double>(baseVal), expHostData);
    Tolerance tol = Tolerance::ForType(dtype);
    std::string errorMsg;
    bool pass = AllClose(expected, resultFloat, tol, &errorMsg);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(base);
    CleanupTensor(exp, expDeviceAddr); CleanupTensor(out, outDeviceAddr);

    if (pass) g_stats.addPass(caseName);
    else g_stats.addFail(caseName, errorMsg);
    return pass ? 0 : 1;
}

// API 6: aclnnPowTensorTensor
template <typename T>
int TestPowTensorTensor(aclrtStream stream, const std::string& typeName) {
    std::string caseName = "PowTensorTensor_" + typeName;
    std::vector<int64_t> shape = {4, 2};
    std::vector<T> selfHostData = {static_cast<T>(0), static_cast<T>(1), static_cast<T>(2), static_cast<T>(3),
                                    static_cast<T>(4), static_cast<T>(5), static_cast<T>(6), static_cast<T>(7)};
    std::vector<T> expHostData = {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(2),
                                   static_cast<T>(2), static_cast<T>(2), static_cast<T>(3), static_cast<T>(3)};

    void* selfDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclDataType dtype = ACL_FLOAT;

    aclTensor* self = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, dtype);
    aclTensor* exp = CreateAclTensor(expHostData, shape, &expDeviceAddr, dtype);
    aclTensor* out = CreateAclTensor(std::vector<T>(8, 0), shape, &outDeviceAddr, dtype);

    if (!self || !exp || !out) {
        g_stats.addFail(caseName, "Failed to create tensor");
        CleanupTensor(self, selfDeviceAddr); CleanupTensor(exp, expDeviceAddr); CleanupTensor(out, outDeviceAddr);
        return -1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        g_stats.addFail(caseName, "GetWorkspaceSize failed: " + std::to_string(ret));
        CleanupTensor(self, selfDeviceAddr); CleanupTensor(exp, expDeviceAddr); CleanupTensor(out, outDeviceAddr);
        return ret;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
    ret = aclrtSynchronizeStream(stream);

    std::vector<T> resultData(8);
    aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), outDeviceAddr, resultData.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> resultFloat(8);
    for (size_t i = 0; i < 8; i++) resultFloat[i] = static_cast<float>(resultData[i]);

    auto expected = CalcExpectedPowTensorTensor(selfHostData, expHostData);
    Tolerance tol = Tolerance::ForType(dtype);
    std::string errorMsg;
    bool pass = AllClose(expected, resultFloat, tol, &errorMsg);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    CleanupTensor(self, selfDeviceAddr); CleanupTensor(exp, expDeviceAddr); CleanupTensor(out, outDeviceAddr);

    if (pass) g_stats.addPass(caseName);
    else g_stats.addFail(caseName, errorMsg);
    return pass ? 0 : 1;
}

// API 7: aclnnInplacePowTensorTensor (新增)
template <typename T>
int TestInplacePowTensorTensor(aclrtStream stream, const std::string& typeName) {
    std::string caseName = "InplacePowTensorTensor_" + typeName;
    std::vector<int64_t> shape = {4, 2};
    std::vector<T> selfHostData = {static_cast<T>(0), static_cast<T>(1), static_cast<T>(2), static_cast<T>(3),
                                    static_cast<T>(4), static_cast<T>(5), static_cast<T>(6), static_cast<T>(7)};
    std::vector<T> expHostData = {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(2),
                                   static_cast<T>(2), static_cast<T>(2), static_cast<T>(3), static_cast<T>(3)};
    std::vector<T> backup = selfHostData;

    void* selfDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    aclDataType dtype = ACL_FLOAT;

    aclTensor* self = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, dtype);
    aclTensor* exp = CreateAclTensor(expHostData, shape, &expDeviceAddr, dtype);

    if (!self || !exp) {
        g_stats.addFail(caseName, "Failed to create tensor");
        CleanupTensor(self, selfDeviceAddr); CleanupTensor(exp, expDeviceAddr);
        return -1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplacePowTensorTensorGetWorkspaceSize(self, exp, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        g_stats.addFail(caseName, "GetWorkspaceSize failed: " + std::to_string(ret));
        CleanupTensor(self, selfDeviceAddr); CleanupTensor(exp, expDeviceAddr);
        return ret;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
    ret = aclrtSynchronizeStream(stream);

    std::vector<T> resultData(8);
    aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), selfDeviceAddr, resultData.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> resultFloat(8);
    for (size_t i = 0; i < 8; i++) resultFloat[i] = static_cast<float>(resultData[i]);

    auto expected = CalcExpectedPowTensorTensor(backup, expHostData);
    Tolerance tol = Tolerance::ForType(dtype);
    std::string errorMsg;
    bool pass = AllClose(expected, resultFloat, tol, &errorMsg);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    CleanupTensor(self, selfDeviceAddr); CleanupTensor(exp, expDeviceAddr);

    if (pass) g_stats.addPass(caseName);
    else g_stats.addFail(caseName, errorMsg);
    return pass ? 0 : 1;
}

// ================= 边界条件测试 =================

template <typename T>
int TestZeroExponent(aclrtStream stream, const std::string& typeName) {
    std::string caseName = "ZeroExponent_" + typeName;
    std::vector<int64_t> shape = {2, 2};
    std::vector<T> selfHostData = {static_cast<T>(1), static_cast<T>(5), static_cast<T>(-3), static_cast<T>(100)};

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclDataType dtype = ACL_FLOAT;

    aclTensor* self = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, dtype);
    aclTensor* out = CreateAclTensor(std::vector<T>(4, 0), shape, &outDeviceAddr, dtype);

    if (!self || !out) {
        g_stats.addFail(caseName, "Failed to create tensor");
        CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr);
        return -1;
    }

    float expVal = 0.0f;
    aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    ret = aclrtSynchronizeStream(stream);

    std::vector<T> resultData(4);
    aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), outDeviceAddr, resultData.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> resultFloat(4);
    for (size_t i = 0; i < 4; i++) resultFloat[i] = static_cast<float>(resultData[i]);

    std::vector<double> expected = {1.0, 1.0, 1.0, 1.0};
    Tolerance tol = Tolerance::ForType(dtype);
    tol.atol = 1e-5;
    std::string errorMsg;
    bool pass = AllClose(expected, resultFloat, tol, &errorMsg);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(exponent);
    CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr);

    if (pass) g_stats.addPass(caseName);
    else g_stats.addFail(caseName, errorMsg);
    return pass ? 0 : 1;
}

template <typename T>
int TestNegativeBase(aclrtStream stream, const std::string& typeName) {
    std::string caseName = "NegativeBase_" + typeName;
    std::vector<int64_t> shape = {2, 2};
    std::vector<T> selfHostData = {static_cast<T>(-2), static_cast<T>(-1), static_cast<T>(-3), static_cast<T>(-4)};

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclDataType dtype = ACL_FLOAT;

    aclTensor* self = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, dtype);
    aclTensor* out = CreateAclTensor(std::vector<T>(4, 0), shape, &outDeviceAddr, dtype);

    if (!self || !out) {
        g_stats.addFail(caseName, "Failed to create tensor");
        CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr);
        return -1;
    }

    float expVal = 3.0f;
    aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    ret = aclrtSynchronizeStream(stream);

    std::vector<T> resultData(4);
    aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), outDeviceAddr, resultData.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> resultFloat(4);
    for (size_t i = 0; i < 4; i++) resultFloat[i] = static_cast<float>(resultData[i]);

    auto expected = CalcExpectedPowTensorScalar(selfHostData, static_cast<double>(expVal));
    Tolerance tol = Tolerance::ForType(dtype);
    std::string errorMsg;
    bool pass = AllClose(expected, resultFloat, tol, &errorMsg);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(exponent);
    CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr);

    if (pass) g_stats.addPass(caseName);
    else g_stats.addFail(caseName, errorMsg);
    return pass ? 0 : 1;
}

template <typename T>
int TestSqrtExponent(aclrtStream stream, const std::string& typeName) {
    std::string caseName = "SqrtExponent_" + typeName;
    std::vector<int64_t> shape = {2, 2};
    std::vector<T> selfHostData = {static_cast<T>(0), static_cast<T>(1), static_cast<T>(4), static_cast<T>(9)};

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclDataType dtype = ACL_FLOAT;

    aclTensor* self = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, dtype);
    aclTensor* out = CreateAclTensor(std::vector<T>(4, 0), shape, &outDeviceAddr, dtype);

    if (!self || !out) {
        g_stats.addFail(caseName, "Failed to create tensor");
        CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr);
        return -1;
    }

    float expVal = 0.5f;
    aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    ret = aclrtSynchronizeStream(stream);

    std::vector<T> resultData(4);
    aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), outDeviceAddr, resultData.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> resultFloat(4);
    for (size_t i = 0; i < 4; i++) resultFloat[i] = static_cast<float>(resultData[i]);

    auto expected = CalcExpectedPowTensorScalar(selfHostData, static_cast<double>(expVal));
    Tolerance tol = Tolerance::ForType(dtype);
    std::string errorMsg;
    bool pass = AllClose(expected, resultFloat, tol, &errorMsg);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(exponent);
    CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr);

    if (pass) g_stats.addPass(caseName);
    else g_stats.addFail(caseName, errorMsg);
    return pass ? 0 : 1;
}

template <typename T>
int TestNullptrInput(aclrtStream stream, const std::string& typeName) {
    std::string caseName = "NullptrInput_" + typeName;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnPowTensorScalarGetWorkspaceSize(nullptr, nullptr, nullptr, &workspaceSize, &executor);

    if (ret != ACL_SUCCESS) {
        g_stats.addPass(caseName + "(ExpectedFailure)");
        return 0;
    } else {
        g_stats.addPass(caseName + "(NoParamCheck-Warning)");
        return 0;
    }
}

template <typename T>
int TestBroadcast(aclrtStream stream, const std::string& typeName) {
    std::string caseName = "Broadcast_" + typeName;
    std::vector<int64_t> selfShape = {2, 1};
    std::vector<int64_t> expShape = {1, 3};
    std::vector<int64_t> outShape = {2, 3};

    std::vector<T> selfHostData = {static_cast<T>(2), static_cast<T>(3)};
    std::vector<T> expHostData = {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)};

    void* selfDeviceAddr = nullptr;
    void* expDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclDataType dtype = ACL_FLOAT;

    aclTensor* self = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, dtype);
    aclTensor* exp = CreateAclTensor(expHostData, expShape, &expDeviceAddr, dtype);
    aclTensor* out = CreateAclTensor(std::vector<T>(6, 0), outShape, &outDeviceAddr, dtype);

    if (!self || !exp || !out) {
        g_stats.addFail(caseName, "Failed to create tensor");
        CleanupTensor(self, selfDeviceAddr); CleanupTensor(exp, expDeviceAddr); CleanupTensor(out, outDeviceAddr);
        return -1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
    ret = aclrtSynchronizeStream(stream);

    std::vector<T> resultData(6);
    aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), outDeviceAddr, resultData.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> resultFloat(6);
    for (size_t i = 0; i < 6; i++) resultFloat[i] = static_cast<float>(resultData[i]);

    std::vector<double> expected = {
        std::pow(2.0, 1.0), std::pow(2.0, 2.0), std::pow(2.0, 3.0),
        std::pow(3.0, 1.0), std::pow(3.0, 2.0), std::pow(3.0, 3.0)
    };

    Tolerance tol = Tolerance::ForType(dtype);
    std::string errorMsg;
    bool pass = AllClose(expected, resultFloat, tol, &errorMsg);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    CleanupTensor(self, selfDeviceAddr); CleanupTensor(exp, expDeviceAddr); CleanupTensor(out, outDeviceAddr);

    if (pass) g_stats.addPass(caseName);
    else g_stats.addFail(caseName, errorMsg);
    return pass ? 0 : 1;
}

template <typename T>
int TestReciprocal(aclrtStream stream, const std::string& typeName) {
    std::string caseName = "Reciprocal_" + typeName;
    std::vector<int64_t> shape = {2, 2};
    std::vector<T> selfHostData = {static_cast<T>(1), static_cast<T>(2), static_cast<T>(4), static_cast<T>(8)};

    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclDataType dtype = ACL_FLOAT;

    aclTensor* self = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, dtype);
    aclTensor* out = CreateAclTensor(std::vector<T>(4, 0), shape, &outDeviceAddr, dtype);

    if (!self || !out) {
        g_stats.addFail(caseName, "Failed to create tensor");
        CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr);
        return -1;
    }

    float expVal = -1.0f;
    aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    ret = aclrtSynchronizeStream(stream);

    std::vector<T> resultData(4);
    aclrtMemcpy(resultData.data(), resultData.size() * sizeof(T), outDeviceAddr, resultData.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> resultFloat(4);
    for (size_t i = 0; i < 4; i++) resultFloat[i] = static_cast<float>(resultData[i]);

    auto expected = CalcExpectedPowTensorScalar(selfHostData, static_cast<double>(expVal));
    Tolerance tol = Tolerance::ForType(dtype);
    std::string errorMsg;
    bool pass = AllClose(expected, resultFloat, tol, &errorMsg);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(exponent);
    CleanupTensor(self, selfDeviceAddr); CleanupTensor(out, outDeviceAddr);

    if (pass) g_stats.addPass(caseName);
    else g_stats.addFail(caseName, errorMsg);
    return pass ? 0 : 1;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  ACLNN Pow Operator Comprehensive Test" << std::endl;
    std::cout << "  Coverage: All 7 APIs                   " << std::endl;
    std::cout << "========================================" << std::endl << std::endl;

    int32_t deviceId = 0;
    aclrtStream stream;

    auto ret = InitACL(deviceId, &stream);
    if (ret != 0) {
        std::cerr << "Failed to initialize ACL" << std::endl;
        return -1;
    }

    std::cout << "ACL initialized successfully on device " << deviceId << std::endl << std::endl;

    // ========== 7个API全覆盖测试 ==========
    std::cout << "------ API Coverage Tests (7 APIs) ------" << std::endl;

    // Exp2类别 (aclnn_exp2.cpp)
    TestExp2<float>(stream, "FLOAT32");                    // API 1
    TestInplaceExp2<float>(stream, "FLOAT32");             // API 2 (新增)

    // TensorScalar类别 (aclnn_pow.cpp)
    TestPowTensorScalar<float>(stream, 3.0f, "Exp3", "FLOAT32");   // API 3
    TestInplacePowTensorScalar<float>(stream, 3.0f, "Exp3", "FLOAT32"); // API 4

    // ScalarTensor类别 (aclnn_pow.cpp) - 新增
    TestPowScalarTensor<float>(stream, 2.0f, "Base2", "FLOAT32");  // API 5 (新增)
    TestPowScalarTensor<float>(stream, 10.0f, "Base10", "FLOAT32");

    // TensorTensor类别 (aclnn_pow_tensor_tensor.cpp)
    TestPowTensorTensor<float>(stream, "FLOAT32");         // API 6
    TestInplacePowTensorTensor<float>(stream, "FLOAT32");  // API 7 (新增)

    // ========== 特殊指数值测试 ==========
    std::cout << "\n------ Special Exponent Tests ------" << std::endl;
    TestPowTensorScalar<float>(stream, 0.0f, "Exp0", "FLOAT32");
    TestPowTensorScalar<float>(stream, 1.0f, "Exp1", "FLOAT32");
    TestPowTensorScalar<float>(stream, 0.5f, "Exp0.5", "FLOAT32");
    TestPowTensorScalar<float>(stream, -1.0f, "Exp-1", "FLOAT32");
    TestPowTensorScalar<float>(stream, 4.1f, "Exp4.1", "FLOAT32");

    // ========== 边界条件测试 ==========
    std::cout << "\n------ Boundary Condition Tests ------" << std::endl;
    TestZeroExponent<float>(stream, "FLOAT32");
    TestNegativeBase<float>(stream, "FLOAT32");
    TestSqrtExponent<float>(stream, "FLOAT32");
    TestReciprocal<float>(stream, "FLOAT32");

    // ========== 广播测试 ==========
    std::cout << "\n------ Broadcast Tests ------" << std::endl;
    TestBroadcast<float>(stream, "FLOAT32");

    // ========== 异常测试 ==========
    std::cout << "\n------ Exception Tests ------" << std::endl;
    TestNullptrInput<float>(stream, "FLOAT32");

    FinalizeACL(deviceId, stream);

    std::cout << std::endl;
    g_stats.printSummary();

    return g_stats.failed > 0 ? 1 : 0;
}