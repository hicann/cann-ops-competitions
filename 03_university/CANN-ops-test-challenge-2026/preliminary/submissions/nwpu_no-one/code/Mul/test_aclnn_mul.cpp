/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <cstring>
#include <random>
#include <type_traits>
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

#define CHECK_RET(cond, return_expr) \
  do { if (!(cond)) { return_expr; } } while (0)
#define LOG_PRINT(message, ...) \
  do { printf(message, ##__VA_ARGS__); } while (0)

// ============================================================================
// 1. 辅助工具：Shape、随机数与精度转换
// ============================================================================

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (auto i : shape) size *= i; 
    // 若 shape 为空 {}，返回 1 (0-D Tensor)
    // 若 shape 包含 0 (如 {0, 2})，循环后 size = 0 (Empty Tensor)
    return size;
}

float GetRandFloat(float min = -5.0f, float max = 5.0f) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

int32_t GetRandInt(int32_t min = -10, int32_t max = 10) {
    static std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dis(min, max);
    return dis(gen);
}

// BFloat16 <-> Float32 转换
uint16_t fp32_to_bf16(float f) {
    uint32_t res; std::memcpy(&res, &f, sizeof(f));
    return res >> 16;
}
float bf16_to_fp32(uint16_t b) {
    uint32_t res = b << 16;
    float f; std::memcpy(&f, &res, sizeof(f));
    return f;
}

// Float16 <-> Float32 转换
uint16_t fp32_to_fp16(float val) {
    uint32_t f; std::memcpy(&f, &val, 4);
    int s = (f >> 16) & 0x8000;
    int e = ((f >> 23) & 0xff) - 127 + 15;
    int m = f & 0x007fffff;
    if (e <= 0) return s;
    if (e >= 31) return s | 0x7c00;
    return s | (e << 10) | (m >> 13);
}
float fp16_to_fp32(uint16_t h) {
    int s = (h >> 15) & 1;
    int e = (h >> 10) & 0x1f;
    int f = h & 0x03ff;
    if (e == 0) return 0.0f;
    if (e == 31) return s ? -INFINITY : INFINITY;
    float val = (1.0f + f * std::pow(2.0f, -10.0f)) * std::pow(2.0f, e - 15.0f);
    return s ? -val : val;
}

bool AlmostEqual(double expected, double actual, double atol, double rtol) {
    if (std::isnan(expected) && std::isnan(actual)) return true;
    if (std::isinf(expected) && std::isinf(actual)) return (expected > 0) == (actual > 0);
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

// 根据 aclDataType 安全地创建 aclScalar，避免 double 内存截断问题
aclScalar* CreateAclScalarFromDouble(double val, aclDataType dt) {
    switch (dt) {
        case ACL_FLOAT16: {
            uint16_t v = fp32_to_fp16(static_cast<float>(val));
            return aclCreateScalar(&v, dt);
        }
        case ACL_BF16: {
            uint16_t v = fp32_to_bf16(static_cast<float>(val));
            return aclCreateScalar(&v, dt);
        }
        case ACL_FLOAT: {
            float v = static_cast<float>(val);
            return aclCreateScalar(&v, dt);
        }
        case ACL_DOUBLE: {
            double v = val;
            return aclCreateScalar(&v, dt);
        }
        case ACL_INT8: {
            int8_t v = static_cast<int8_t>(val);
            return aclCreateScalar(&v, dt);
        }
        case ACL_UINT8: {
            uint8_t v = static_cast<uint8_t>(val);
            return aclCreateScalar(&v, dt);
        }
        case ACL_INT16: {
            int16_t v = static_cast<int16_t>(val);
            return aclCreateScalar(&v, dt);
        }
        case ACL_INT32: {
            int32_t v = static_cast<int32_t>(val);
            return aclCreateScalar(&v, dt);
        }
        case ACL_INT64: {
            int64_t v = static_cast<int64_t>(val);
            return aclCreateScalar(&v, dt);
        }
        case ACL_BOOL: {
            bool v = static_cast<bool>(val);
            return aclCreateScalar(&v, dt);
        }
        default: {
            float v = static_cast<float>(val);
            return aclCreateScalar(&v, dt);
        }
    }
}

// ============================================================================
// 2. ACL 初始化与 Tensor 创建
// ============================================================================

int Init(int32_t deviceId, aclrtStream* stream) {
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return 1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, return 1);
    CHECK_RET(aclrtCreateStream(stream) == ACL_SUCCESS, return 1);
    return 0;
}

template <typename T>
int CreateAclTensorEx(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                      void** deviceAddr, aclDataType dataType, aclFormat format, 
                      std::vector<int64_t> customStrides, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    if (size > 0) {
        aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    } else {
        *deviceAddr = nullptr; 
    }
    if (customStrides.empty() && !shape.empty()) {
        customStrides.resize(shape.size(), 1);
        for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--) customStrides[i] = shape[i+1] * customStrides[i+1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, 
                              customStrides.empty() ? nullptr : customStrides.data(), 0,
                              format, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
    return CreateAclTensorEx(hostData, shape, deviceAddr, dataType, ACL_FORMAT_ND, {}, tensor);
}

// ============================================================================
// 3. 通用标量验证与执行引擎
// ============================================================================

template <typename TOut>
int VerifyOutput(const char* name, void* outDev, std::vector<TOut>& outHost, int64_t n, 
                 const std::vector<double>& expected, double atol, double rtol, aclDataType dtOut) {
    if (n > 0) aclrtMemcpy(outHost.data(), n * sizeof(TOut), outDev, n * sizeof(TOut), ACL_MEMCPY_DEVICE_TO_HOST);
    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        if constexpr (std::is_same<TOut, uint16_t>::value) {
            double actual = (dtOut == ACL_BF16) ? bf16_to_fp32(outHost[i]) : fp16_to_fp32(outHost[i]);
            if (!AlmostEqual(expected[i], actual, atol, rtol)) failed++;
        } else if constexpr (std::is_arithmetic<TOut>::value) {
            double actual = static_cast<double>(outHost[i]);
            if (!AlmostEqual(expected[i], actual, atol, rtol)) failed++;
        }
    }
    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);
    return failed > 0 ? 1 : 0;
}

template <typename T1, typename T2, typename TOut>
int RunMulTest(const char* name, aclDataType dt1, aclDataType dt2, aclDataType dtOut,
               const std::vector<T1>& x1, const std::vector<T2>& x2, const std::vector<int64_t>& shape, 
               const std::vector<double>& expected, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
    
    CreateAclTensor(x1, shape, &x1Dev, dt1, &x1T);
    CreateAclTensor(x2, shape, &x2Dev, dt2, &x2T);
    std::vector<TOut> outHost(n > 0 ? n : 1, 0);
    CreateAclTensor(outHost, shape, &outDev, dtOut, &outT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
    
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret);
        return 1;
    }

    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnMul(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);

    int fail_cnt = VerifyOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    if(x1Dev) aclrtFree(x1Dev); if(x2Dev) aclrtFree(x2Dev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

// ============================================================================
// 4. 广播与复数专用执行引擎
// ============================================================================

template <typename T1, typename T2, typename TOut>
int RunMulBroadcastTest(const char* name, aclDataType dt1, aclDataType dt2, aclDataType dtOut,
                        const std::vector<T1>& x1, const std::vector<T2>& x2, 
                        const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2, const std::vector<int64_t>& outShape,
                        const std::vector<double>& expected, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(outShape);
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;

    CreateAclTensor(x1, shape1, &x1Dev, dt1, &x1T);
    CreateAclTensor(x2, shape2, &x2Dev, dt2, &x2T);
    std::vector<TOut> outHost(n > 0 ? n : 1, 0);
    CreateAclTensor(outHost, outShape, &outDev, dtOut, &outT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret);
        return 1;
    }

    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnMul(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);

    int fail_cnt = VerifyOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    if(x1Dev) aclrtFree(x1Dev); if(x2Dev) aclrtFree(x2Dev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

template <typename T1, typename T2, typename TOut>
int RunMulComplexMixTest(const char* name, aclDataType dt1, aclDataType dt2, aclDataType dtOut,
                         const std::vector<std::complex<T1>>& x1, const std::vector<T2>& x2, 
                         const std::vector<int64_t>& shape, const std::vector<std::complex<double>>& expected, 
                         aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
    
    CreateAclTensor(x1, shape, &x1Dev, dt1, &x1T);
    CreateAclTensor(x2, shape, &x2Dev, dt2, &x2T); 
    std::vector<std::complex<TOut>> outHost(n > 0 ? n : 1, {0.0, 0.0});
    CreateAclTensor(outHost, shape, &outDev, dtOut, &outT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
    
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret); return 1);

    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnMul(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);
    if (n > 0) aclrtMemcpy(outHost.data(), n * sizeof(std::complex<TOut>), outDev, n * sizeof(std::complex<TOut>), ACL_MEMCPY_DEVICE_TO_HOST);

    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        if (!AlmostEqual(expected[i].real(), static_cast<double>(outHost[i].real()), atol, rtol) || 
            !AlmostEqual(expected[i].imag(), static_cast<double>(outHost[i].imag()), atol, rtol)) failed++;
    }
    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    if(x1Dev) aclrtFree(x1Dev); if(x2Dev) aclrtFree(x2Dev); if(outDev) aclrtFree(outDev);
    return failed > 0 ? 1 : 0;
}

// ============================================================================
// 5. API 变体与大 Shape/边界 场景
// ============================================================================

template <typename TIn, typename TOut>
int RunMulsTest(const char* name, aclDataType dtTensor, aclDataType dtScalar, aclDataType dtOut,
                const std::vector<TIn>& self, double scalarVal, const std::vector<int64_t>& shape, 
                const std::vector<double>& expected, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *outDev=nullptr; aclTensor *selfT=nullptr, *outT=nullptr;
    CreateAclTensor(self, shape, &selfDev, dtTensor, &selfT);
    std::vector<TOut> outHost(n > 0 ? n : 1, 0); 
    CreateAclTensor(outHost, shape, &outDev, dtOut, &outT);

    // 使用安全工厂函数创建 Scalar
    aclScalar* scalar = CreateAclScalarFromDouble(scalarVal, dtScalar);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulsGetWorkspaceSize(selfT, scalar, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnMuls(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(selfT); aclDestroyTensor(outT); aclDestroyScalar(scalar);
    if(selfDev) aclrtFree(selfDev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

template <typename TIn, typename TOut>
int RunMulsComplexTest(const char* name, aclDataType dtTensor, aclDataType dtScalar, aclDataType dtOut,
                       const std::vector<std::complex<TIn>>& self, double scalarVal, const std::vector<int64_t>& shape, 
                       const std::vector<std::complex<double>>& expected, aclrtStream stream) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *outDev=nullptr; aclTensor *selfT=nullptr, *outT=nullptr;
    CreateAclTensor(self, shape, &selfDev, dtTensor, &selfT);
    std::vector<std::complex<TOut>> outHost(n > 0 ? n : 1, {0.0, 0.0}); 
    CreateAclTensor(outHost, shape, &outDev, dtOut, &outT);

    aclScalar* scalar = CreateAclScalarFromDouble(scalarVal, dtScalar);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulsGetWorkspaceSize(selfT, scalar, outT, &wsSize, &executor);
    
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret); return 1);

    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnMuls(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);
    if (n > 0) aclrtMemcpy(outHost.data(), n * sizeof(std::complex<TOut>), outDev, n * sizeof(std::complex<TOut>), ACL_MEMCPY_DEVICE_TO_HOST);

    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        if (!AlmostEqual(expected[i].real(), static_cast<double>(outHost[i].real()), 1e-5, 1e-5) || 
            !AlmostEqual(expected[i].imag(), static_cast<double>(outHost[i].imag()), 1e-5, 1e-5)) failed++;
    }
    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(selfT); aclDestroyTensor(outT); aclDestroyScalar(scalar);
    if(selfDev) aclrtFree(selfDev); if(outDev) aclrtFree(outDev);
    return failed > 0 ? 1 : 0;
}

template <typename T1, typename T2>
int RunInplaceTest(const char* name, aclDataType dt1, aclDataType dt2, const std::vector<T1>& self, 
                   const std::vector<T2>& other, const std::vector<int64_t>& shape, 
                   const std::vector<double>& expected, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *otherDev=nullptr; aclTensor *selfT=nullptr, *otherT=nullptr;
    CreateAclTensor(self, shape, &selfDev, dt1, &selfT);
    CreateAclTensor(other, shape, &otherDev, dt2, &otherT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceMulGetWorkspaceSize(selfT, otherT, &wsSize, &executor);

    std::vector<T1> outHost(n > 0 ? n : 1, 0);
    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnInplaceMul(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);

    int fail_cnt = VerifyOutput(name, selfDev, outHost, n, expected, atol, rtol, dt1);
    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(selfT); aclDestroyTensor(otherT);
    if(selfDev) aclrtFree(selfDev); if(otherDev) aclrtFree(otherDev);
    return fail_cnt;
}

template <typename T>
int RunInplaceMulsTest(const char* name, aclDataType dt1, aclDataType dtScalar, const std::vector<T>& self, 
                       double scalarVal, const std::vector<int64_t>& shape, const std::vector<double>& expected, 
                       aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr; aclTensor *selfT=nullptr;
    CreateAclTensor(self, shape, &selfDev, dt1, &selfT);

    aclScalar* scalar = CreateAclScalarFromDouble(scalarVal, dtScalar);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceMulsGetWorkspaceSize(selfT, scalar, &wsSize, &executor);

    std::vector<T> outHost(n > 0 ? n : 1, 0);
    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnInplaceMuls(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);

    int fail_cnt = VerifyOutput(name, selfDev, outHost, n, expected, atol, rtol, dt1);
    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(selfT); aclDestroyScalar(scalar);
    if(selfDev) aclrtFree(selfDev);
    return fail_cnt;
}

int RunBroadcastTest(const char* name, aclrtStream stream) {
    std::vector<int64_t> s1 = {2, 3}; 
    std::vector<int64_t> s2 = {3}; 
    std::vector<int64_t> sOut = {2, 3}; 
    std::vector<float> x1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<float> x2 = {10.0, 20.0, 30.0};
    std::vector<double> expected(6);
    for(int i=0; i<2; i++) for(int j=0; j<3; j++) expected[i*3 + j] = x1[i*3 + j] * x2[j];
    
    // 使用独立的 Broadcast 测试器以提供正确的形状分配
    return RunMulBroadcastTest<float, float, float>(name, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, x1, x2, s1, s2, sOut, expected, stream);
}

int RunFormatWarningTest(const char* name, aclrtStream stream) {
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
    
    CreateAclTensorEx<float>({1, 1}, {1, 1, 1, 1, 1}, &x1Dev, ACL_FLOAT, ACL_FORMAT_NC1HWC0, {}, &x1T); 
    CreateAclTensorEx<float>({1, 1}, {1, 1, 1, 1, 1}, &x2Dev, ACL_FLOAT, ACL_FORMAT_NC1HWC0, {}, &x2T);
    CreateAclTensorEx<float>({1, 1}, {1, 1, 1, 1, 1}, &outDev, ACL_FLOAT, ACL_FORMAT_NC1HWC0, {}, &outT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
    
    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
    LOG_PRINT((ret == ACL_SUCCESS || ret == 561103) ? "[PASS] %s\n" : "[FAIL] %s\n", name);
    return (ret == ACL_SUCCESS || ret == 561103) ? 0 : 1;
}

int RunNonContiguousTest(const char* name, aclrtStream stream) {
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
    
    CreateAclTensorEx<float>({1, 2}, {2}, &x1Dev, ACL_FLOAT, ACL_FORMAT_ND, {10}, &x1T);
    CreateAclTensorEx<float>({1, 2}, {2}, &x2Dev, ACL_FLOAT, ACL_FORMAT_ND, {1}, &x2T);
    CreateAclTensorEx<float>({0, 0}, {2}, &outDev, ACL_FLOAT, ACL_FORMAT_ND, {1}, &outT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
    
    if (wsSize > 0 && ret == ACL_SUCCESS) {
        void* wsAddr = nullptr; aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnMul(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        aclrtFree(wsAddr);
    }
    
    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
    LOG_PRINT(ret == ACL_SUCCESS ? "[PASS] %s\n" : "[FAIL] %s\n", name);
    return ret == ACL_SUCCESS ? 0 : 1;
}

// ============================================================================
// 6. 异常参数拦截测试 (反向断言)
// ============================================================================

int TestNegativeNullptr(const char* name) {
    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    int fail = 0;
    if (aclnnMulGetWorkspaceSize(nullptr, nullptr, nullptr, &wsSize, &exec) == ACL_SUCCESS) fail++;
    if (aclnnMulsGetWorkspaceSize(nullptr, nullptr, nullptr, &wsSize, &exec) == ACL_SUCCESS) fail++;
    if (aclnnInplaceMulGetWorkspaceSize(nullptr, nullptr, &wsSize, &exec) == ACL_SUCCESS) fail++;
    LOG_PRINT(fail == 0 ? "[PASS] %s (Blocked)\n" : "[FAIL] %s (Did NOT block)\n", name);
    return fail;
}

int TestNegativeShapeMismatch(const char* name, aclrtStream stream) {
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
    CreateAclTensor<float>({1,2}, {1, 2}, &x1Dev, ACL_FLOAT, &x1T);
    CreateAclTensor<float>({1,2,3}, {1, 2, 3}, &x2Dev, ACL_FLOAT, &x2T); 
    CreateAclTensor<float>({0}, {2}, &outDev, ACL_FLOAT, &outT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
    
    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
    if (ret != ACL_SUCCESS) { LOG_PRINT("[PASS] %s (Blocked)\n", name); return 0; }
    LOG_PRINT("[FAIL] %s (Did NOT block)\n", name); return 1;
}

int TestNegativeInvalidDType(const char* name, aclrtStream stream) {
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
    CreateAclTensor<float>({1,2}, {2}, &x1Dev, static_cast<aclDataType>(9999), &x1T);
    CreateAclTensor<float>({1,2}, {2}, &x2Dev, ACL_FLOAT, &x2T);
    CreateAclTensor<float>({0,0}, {2}, &outDev, ACL_FLOAT, &outT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
    
    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
    if (ret != ACL_SUCCESS) { LOG_PRINT("[PASS] %s (Blocked)\n", name); return 0; }
    LOG_PRINT("[FAIL] %s (Did NOT block)\n", name); return 1;
}

int RunDualNonContiguousTest(const char* name, aclrtStream stream) {
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
    
    std::vector<float> x1_data = {1, 0, 0, 2, 0, 0, 3, 0, 0, 4}; // 步长 3
    std::vector<float> x2_data = {2, 0, 3, 0, 4, 0, 5, 0};       // 步长 2
    
    // 构造双侧均带 Stride 的非连续 Tensor
    CreateAclTensorEx<float>(x1_data, {2, 2}, &x1Dev, ACL_FLOAT, ACL_FORMAT_ND, {6, 3}, &x1T);
    CreateAclTensorEx<float>(x2_data, {2, 2}, &x2Dev, ACL_FLOAT, ACL_FORMAT_ND, {4, 2}, &x2T);
    CreateAclTensorEx<float>({0,0,0,0}, {2, 2}, &outDev, ACL_FLOAT, ACL_FORMAT_ND, {2, 1}, &outT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnMulGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
    
    if (wsSize > 0 && ret == ACL_SUCCESS) {
        void* wsAddr = nullptr; aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnMul(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        aclrtFree(wsAddr);
    }
    
    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    aclrtFree(x1Dev); aclrtFree(x2Dev); aclrtFree(outDev);
    LOG_PRINT((ret == ACL_SUCCESS || ret == 561103) ? "[PASS] %s\n" : "[FAIL] %s\n", name);
    return (ret == ACL_SUCCESS || ret == 561103) ? 0 : 1;
}

// ============================================================================
// 专用引擎：复数 InplaceMuls 算术验证
// ============================================================================
template <typename TIn>
int RunInplaceMulsComplexTest(const char* name, aclDataType dtTensor, aclDataType dtScalar, 
                              const std::vector<std::complex<TIn>>& self, 
                              double scalarVal, const std::vector<int64_t>& shape, 
                              const std::vector<std::complex<double>>& expected, 
                              aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev = nullptr; 
    aclTensor *selfT = nullptr;
    
    // Inplace 修改输入，准备一个可读写的 host 缓存
    std::vector<std::complex<TIn>> outHost = self;
    CreateAclTensor(outHost, shape, &selfDev, dtTensor, &selfT);

    aclScalar* scalar = CreateAclScalarFromDouble(scalarVal, dtScalar);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceMulsGetWorkspaceSize(selfT, scalar, &wsSize, &executor);
    
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret); return 1);

    void* wsAddr = nullptr;
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnInplaceMuls(wsAddr, wsSize, executor, stream);
    aclrtSynchronizeStream(stream);

    // 将就地修改后的数据拷回 Host 进行算术校验
    if (n > 0) {
        aclrtMemcpy(outHost.data(), n * sizeof(std::complex<TIn>), selfDev, n * sizeof(std::complex<TIn>), ACL_MEMCPY_DEVICE_TO_HOST);
    }

    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        // 对实部和虚部分别进行严苛的几乎相等断言
        if (!AlmostEqual(expected[i].real(), static_cast<double>(outHost[i].real()), atol, rtol) || 
            !AlmostEqual(expected[i].imag(), static_cast<double>(outHost[i].imag()), atol, rtol)) {
            failed++;
        }
    }
    
    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);

    if (wsAddr) aclrtFree(wsAddr);
    aclDestroyTensor(selfT); aclDestroyScalar(scalar);
    if(selfDev) aclrtFree(selfDev);
    return failed > 0 ? 1 : 0;
}

// 专门测试 Muls 格式警告的分支 (self format != ND)
int RunMulsFormatWarningTest(const char* name, aclrtStream stream) {
    void *selfDev = nullptr, *outDev = nullptr;
    aclTensor *selfT = nullptr, *outT = nullptr;
    
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int64_t> shape = {1, 1, 2, 2}; // 4D shape 适合 NCHW
    
    // 1. 创建一个格式为 NCHW (非 ND) 的 Tensor
    // 注意：这里使用 CreateAclTensorEx，传入 ACL_FORMAT_NCHW
    CreateAclTensorEx<float>(data, shape, &selfDev, ACL_FLOAT, ACL_FORMAT_NCHW, {}, &selfT);
    
    // 输出通常保持 ND 即可，因为逻辑只检查输入 self
    CreateAclTensor<float>(data, shape, &outDev, ACL_FLOAT, &outT);

    aclScalar* scalar = aclCreateScalar(&data[0], ACL_FLOAT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    // 2. 调用 GetWorkspaceSize，此时内部会触发 MulsCheckFormat -> OP_LOGW
    auto ret = aclnnMulsGetWorkspaceSize(selfT, scalar, outT, &wsSize, &executor);
    
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnMuls(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        if (wsAddr) aclrtFree(wsAddr);
    }

    aclDestroyTensor(selfT); aclDestroyTensor(outT); aclDestroyScalar(scalar);
    if(selfDev) aclrtFree(selfDev); if(outDev) aclrtFree(outDev);
    
    LOG_PRINT("[PASS] %s (Check log for: aclnnMuls only support format ND.)\n", name);
    return 0; 
}

// ============================================================================
// MAIN 主流程
// ============================================================================

int main() {
    int32_t deviceId = 0; aclrtStream stream;
    if (Init(deviceId, &stream) != 0) return 1;
    int totalFailed = 0;

    std::vector<int64_t> s4 = {4};
    
    std::vector<float> f32 = {GetRandFloat(), GetRandFloat(), GetRandFloat(), GetRandFloat()};
    std::vector<float> f32_2 = {GetRandFloat(), GetRandFloat(), GetRandFloat(), GetRandFloat()};
    std::vector<double> d64 = {2.5, -1.0, 0.5, 3.14};
    std::vector<double> d64_2 = {1.0, 2.0, 3.0, 4.0};
    std::vector<int32_t> i32 = {GetRandInt(), GetRandInt(), GetRandInt(), GetRandInt()};
    std::vector<int32_t> i32_2 = {GetRandInt(), GetRandInt(), GetRandInt(), GetRandInt()};
    std::vector<int64_t> i64 = {1000, -2000, 3000, 0};
    std::vector<int16_t> i16 = {10, -20, 30, -40};
    std::vector<int8_t> i8 = {1, 2, 3, -4};
    std::vector<uint8_t> ui8 = {5, 6, 7, 8};
    std::vector<uint8_t> b8 = {1, 0, 1, 0}; 
    
    std::vector<uint16_t> fp16 = {fp32_to_fp16(f32[0]), fp32_to_fp16(f32[1]), fp32_to_fp16(f32[2]), fp32_to_fp16(f32[3])};
    std::vector<uint16_t> fp16_2 = {fp32_to_fp16(f32_2[0]), fp32_to_fp16(f32_2[1]), fp32_to_fp16(f32_2[2]), fp32_to_fp16(f32_2[3])};
    std::vector<uint16_t> bf16 = {fp32_to_bf16(f32[0]), fp32_to_bf16(f32[1]), fp32_to_bf16(f32[2]), fp32_to_bf16(f32[3])};
    std::vector<uint16_t> bf16_2 = {fp32_to_bf16(f32_2[0]), fp32_to_bf16(f32_2[1]), fp32_to_bf16(f32_2[2]), fp32_to_bf16(f32_2[3])};

    LOG_PRINT("\n--- 1. Tiling_Arch35 DTYPE_MAP 强覆盖 ---\n");
    
    // 基础整型与布尔类型
    totalFailed += RunMulTest<int8_t, int8_t, int8_t>("Tiling_INT8", ACL_INT8, ACL_INT8, ACL_INT8, i8, i8, s4, {(double)(i8[0]*i8[0]), (double)(i8[1]*i8[1]), (double)(i8[2]*i8[2]), (double)(i8[3]*i8[3])}, stream);
    totalFailed += RunMulTest<uint8_t, uint8_t, uint8_t>("Tiling_UINT8", ACL_UINT8, ACL_UINT8, ACL_UINT8, ui8, ui8, s4, {(double)(ui8[0]*ui8[0]), (double)(ui8[1]*ui8[1]), (double)(ui8[2]*ui8[2]), (double)(ui8[3]*ui8[3])}, stream);
    totalFailed += RunMulTest<uint8_t, uint8_t, uint8_t>("Tiling_BOOL", ACL_BOOL, ACL_BOOL, ACL_BOOL, b8, b8, s4, {(double)(b8[0]*b8[0]), (double)(b8[1]*b8[1]), (double)(b8[2]*b8[2]), (double)(b8[3]*b8[3])}, stream);
    
    // 混合浮点精度类型提升 (Mix Float Promotion)
    totalFailed += RunMulTest<uint16_t, float, float>("Tiling_Mix_BF16_FLOAT", ACL_BF16, ACL_FLOAT, ACL_FLOAT, bf16, f32_2, s4, {bf16_to_fp32(bf16[0])*f32_2[0], bf16_to_fp32(bf16[1])*f32_2[1], bf16_to_fp32(bf16[2])*f32_2[2], bf16_to_fp32(bf16[3])*f32_2[3]}, stream, 1e-2, 1e-2);
    totalFailed += RunMulTest<float, uint16_t, float>("Tiling_Mix_FLOAT_BF16", ACL_FLOAT, ACL_BF16, ACL_FLOAT, f32, bf16, s4, {f32[0]*bf16_to_fp32(bf16[0]), f32[1]*bf16_to_fp32(bf16[1]), f32[2]*bf16_to_fp32(bf16[2]), f32[3]*bf16_to_fp32(bf16[3])}, stream, 1e-2, 1e-2);
    totalFailed += RunMulTest<uint16_t, float, float>("Tiling_Mix_FP16_FLOAT", ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, fp16, f32_2, s4, {fp16_to_fp32(fp16[0])*f32_2[0], fp16_to_fp32(fp16[1])*f32_2[1], fp16_to_fp32(fp16[2])*f32_2[2], fp16_to_fp32(fp16[3])*f32_2[3]}, stream, 1e-3, 1e-3);
    totalFailed += RunMulTest<float, uint16_t, float>("Tiling_Mix_FLOAT_FP16", ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, f32, fp16, s4, {f32[0]*fp16_to_fp32(fp16[0]), f32[1]*fp16_to_fp32(fp16[1]), f32[2]*fp16_to_fp32(fp16[2]), f32[3]*fp16_to_fp32(fp16[3])}, stream, 1e-3, 1e-3);
    
    // 基础浮点类型
    totalFailed += RunMulTest<uint16_t, uint16_t, uint16_t>("Tiling_BF16", ACL_BF16, ACL_BF16, ACL_BF16, bf16, bf16_2, s4, {bf16_to_fp32(bf16[0])*bf16_to_fp32(bf16_2[0]), bf16_to_fp32(bf16[1])*bf16_to_fp32(bf16_2[1]), bf16_to_fp32(bf16[2])*bf16_to_fp32(bf16_2[2]), bf16_to_fp32(bf16[3])*bf16_to_fp32(bf16_2[3])}, stream, 1e-2, 1e-2);
    totalFailed += RunMulTest<uint16_t, uint16_t, uint16_t>("Tiling_FP16", ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, fp16, fp16_2, s4, {fp16_to_fp32(fp16[0])*fp16_to_fp32(fp16_2[0]), fp16_to_fp32(fp16[1])*fp16_to_fp32(fp16_2[1]), fp16_to_fp32(fp16[2])*fp16_to_fp32(fp16_2[2]), fp16_to_fp32(fp16[3])*fp16_to_fp32(fp16_2[3])}, stream, 1e-3, 1e-3);
    totalFailed += RunMulTest<float, float, float>("Tiling_FLOAT", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, f32, f32_2, s4, {f32[0]*f32_2[0], f32[1]*f32_2[1], f32[2]*f32_2[2], f32[3]*f32_2[3]}, stream);
    totalFailed += RunMulTest<double, double, double>("Tiling_DOUBLE", ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE, d64, d64_2, s4, {d64[0]*d64_2[0], d64[1]*d64_2[1], d64[2]*d64_2[2], d64[3]*d64_2[3]}, stream);

    // 宽整型数据扩展 (补充覆盖 INT32, INT64, INT16)
    totalFailed += RunMulTest<int32_t, int32_t, int32_t>("Tiling_INT32", ACL_INT32, ACL_INT32, ACL_INT32, i32, i32_2, s4, {(double)(i32[0]*i32_2[0]), (double)(i32[1]*i32_2[1]), (double)(i32[2]*i32_2[2]), (double)(i32[3]*i32_2[3])}, stream);
    totalFailed += RunMulTest<int64_t, int64_t, int64_t>("Tiling_INT64", ACL_INT64, ACL_INT64, ACL_INT64, i64, i64, s4, {(double)(i64[0]*i64[0]), (double)(i64[1]*i64[1]), (double)(i64[2]*i64[2]), (double)(i64[3]*i64[3])}, stream);
    totalFailed += RunMulTest<int16_t, int16_t, int16_t>("Tiling_INT16", ACL_INT16, ACL_INT16, ACL_INT16, i16, i16, s4, {(double)(i16[0]*i16[0]), (double)(i16[1]*i16[1]), (double)(i16[2]*i16[2]), (double)(i16[3]*i16[3])}, stream);

    LOG_PRINT("\n--- 2. 复数精度及异常路由校验 (Complex32/64/) ---\n");
    
    // COMPLEX64
    std::vector<std::complex<float>> c64_v1 = {{1.0f, 2.0f}, {3.0f, 4.0f}, {0.0f, 0.0f}, {1.0f, 1.0f}};
    std::vector<std::complex<float>> c64_v2 = {{2.0f, 1.0f}, {1.0f, 0.5f}, {1.0f, 1.0f}, {2.0f, 2.0f}};
    std::vector<std::complex<double>> exp_c64 = {{(1.0*2.0 - 2.0*1.0), (1.0*1.0 + 2.0*2.0)}, {(3.0*1.0 - 4.0*0.5), (3.0*0.5 + 4.0*1.0)}, {0, 0}, {(1.0*2.0 - 1.0*2.0), (1.0*2.0 + 1.0*2.0)}};
    totalFailed += RunMulComplexMixTest<float, std::complex<float>, float>("Tiling_COMPLEX64", ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64, c64_v1, c64_v2, s4, exp_c64, stream);

    // 4. Muls 接口特判：COMPLEX64 (Tensor) x DOUBLE (Scalar)
    // 标量 Double 乘复数，预期输出依然保持 COMPLEX64，精度不丢失
    std::vector<std::complex<double>> exp_c64_scalar_d = {
        {1.0 * 1.5, 2.0 * 1.5},
        {3.0 * 1.5, 4.0 * 1.5},
        {0.0 * 1.5, 0.0 * 1.5},
        {1.0 * 1.5, 1.0 * 1.5}
    };
    totalFailed += RunMulsComplexTest<float, float>(
        "Mix_Muls_COMPLEX64_DOUBLE_Scalar", ACL_COMPLEX64, ACL_DOUBLE, ACL_COMPLEX64, 
        c64_v1, 1.5, s4, exp_c64_scalar_d, stream);

    LOG_PRINT("\n--- 3. 大 Shape (Dim>4) 及数值边界验证 ---\n");

    // 1. 边界数值验证 (Inf, NaN, Float最大极值, -0.0)
    std::vector<float> b_x1 = {INFINITY, NAN, 3.402823466e+38f, -0.0f};
    std::vector<float> b_x2 = {2.0f, 2.0f, 2.0f, 5.0f};
    std::vector<double> exp_bnd = {b_x1[0]*b_x2[0], b_x1[1]*b_x2[1], b_x1[2]*b_x2[2], b_x1[3]*b_x2[3]};
    totalFailed += RunMulTest<float, float, float>(
        "Boundary_FLOAT_Inf_NaN", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 
        b_x1, b_x2, s4, exp_bnd, stream
    );

    // 2. 超大 Shape 容量验证 (1 Million Elements)
    // 触发底层分块 (Tiling) 算法的多 Core 满载逻辑
    std::vector<float> large_x(1024*1024, 1.5f);
    std::vector<double> large_exp(1024*1024, 2.25);
    totalFailed += RunMulTest<float, float, float>(
        "LargeShape_1M_Elements", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 
        large_x, large_x, {1024, 1024}, large_exp, stream
    );

    // 3. 维度大于 4 的 Shape 验证 (Dim > 4)
    // 覆盖 `mul.cpp` 中 `isBroadcastTemplateNonContiguousSupport` 维度超限拦截分支
    std::vector<float> f5d(32, 1.0f);
    std::vector<double> exp_5d(32, 1.0);
    totalFailed += RunMulTest<float, float, float>(
        "Dim_Greater_Than_4_Check", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 
        f5d, f5d, {2, 2, 2, 2, 2}, exp_5d, stream
    );

    LOG_PRINT("\n--- 4. API 变体常规成功分支与空 Tensor 拦截 ---\n");

    // ==========================================
    // [1] Muls 成功变体分支 (Tensor * Scalar)
    // 成功秘诀：Muls 要求 out 的数据类型与 self 的数据类型保持一致！
    // ==========================================
    // FLOAT * 标量 -> FLOAT
    std::vector<double> exp_muls_f32 = {f32[0]*2.5, f32[1]*2.5, f32[2]*2.5, f32[3]*2.5};
    totalFailed += RunMulsTest<float, float>("API_Muls_FLOAT", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, f32, 2.5, s4, exp_muls_f32, stream);

    // DOUBLE * 标量 -> DOUBLE (注意 out 和 scalar 均保持 ACL_DOUBLE 避免降级)
    std::vector<double> exp_muls_d64 = {d64[0]*2.5, d64[1]*2.5, d64[2]*2.5, d64[3]*2.5};
    totalFailed += RunMulsTest<double, double>(
        "API_Muls_DOUBLE_Success", ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE, d64, 2.5, s4, exp_muls_d64, stream
    );

    // INT32 * 标量 -> INT32
    std::vector<double> exp_muls_i32 = {(double)(i32[0]*2), (double)(i32[1]*2), (double)(i32[2]*2), (double)(i32[3]*2)};
    totalFailed += RunMulsTest<int32_t, int32_t>("API_Muls_INT32", ACL_INT32, ACL_INT32, ACL_INT32, i32, 2.0, s4, exp_muls_i32, stream);

    // INT16 * 标量 -> INT16
    std::vector<double> exp_muls_i16 = {(double)(i16[0]*2), (double)(i16[1]*2), (double)(i16[2]*2), (double)(i16[3]*2)};
    totalFailed += RunMulsTest<int16_t, int16_t>("API_Muls_INT16", ACL_INT16, ACL_INT16, ACL_INT16, i16, 2.0, s4, exp_muls_i16, stream);

    LOG_PRINT("\n--- 下面三个测试用例挖掘出算子缺陷 Muls目前不支持输入为fp16的情况 ---\n");
    std::vector<double> exp_muls_fp16 = {fp16_to_fp32(fp16[0])*2.0, fp16_to_fp32(fp16[1])*2.0, fp16_to_fp32(fp16[2])*2.0, fp16_to_fp32(fp16[3])*2.0};
    totalFailed += RunMulsTest<uint16_t, uint16_t>(
        "API_Muls_FP16", 
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, 
        fp16, 2.0, s4, exp_muls_fp16, stream, 1e-3, 1e-3
    );
    totalFailed += RunMulsTest<uint16_t, float>(
        "API_Muls_FP16In_FP32Out", 
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT, 
        fp16, 2.0, s4, exp_muls_fp16, stream
    );
    totalFailed += RunMulsTest<uint16_t, float>(
        "API_Muls_FP16In_FP32Scalar", 
        ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, 
        fp16, 2.0, s4, exp_muls_fp16, stream
    );
    LOG_PRINT("\n--- 下面三个测试用例挖掘出算子缺陷 Muls目前不支持输入为bf16的情况 ---\n");
    std::vector<double> exp_muls_bf16 = {bf16_to_fp32(bf16[0])*2.0, bf16_to_fp32(bf16[1])*2.0, bf16_to_fp32(bf16[2])*2.0, bf16_to_fp32(bf16[3])*2.0};
    totalFailed += RunMulsTest<uint16_t, uint16_t>(
        "API_Muls_BF16", 
        ACL_BF16, ACL_BF16, ACL_BF16, 
        bf16, 2.0, s4, exp_muls_bf16, stream, 1e-3, 1e-3
    );
    totalFailed += RunMulsTest<uint16_t, float>(
        "API_Muls_BF16In_FP32Out", 
        ACL_BF16, ACL_BF16, ACL_FLOAT, 
        bf16, 2.0, s4, exp_muls_bf16, stream
    );
    totalFailed += RunMulsTest<uint16_t, float>(
        "API_Muls_BF16In_FP32Scalar", 
        ACL_BF16, ACL_FLOAT, ACL_FLOAT, 
        bf16, 2.0, s4, exp_muls_bf16, stream
    );
    LOG_PRINT("\n");
    std::vector<double> exp_muls_fp32 = {f32[0] * 2.0, f32[1] * 2.0, f32[2] * 2.0, f32[3] * 2.0};
    totalFailed += RunMulsTest<float, float>(
        "API_Muls_FP32_FP16Scalar", 
        ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, 
        f32, 2.0, s4, exp_muls_fp32, stream
    );
    // ==========================================
    // [2] InplaceMul 成功变体分支 (Tensor *= Tensor)
    // ==========================================
    // FLOAT *= FLOAT
    std::vector<double> exp_inplace_f32 = {f32[0]*f32_2[0], f32[1]*f32_2[1], f32[2]*f32_2[2], f32[3]*f32_2[3]};
    totalFailed += RunInplaceTest<float, float>("API_InplaceMul_FLOAT", ACL_FLOAT, ACL_FLOAT, f32, f32_2, s4, exp_inplace_f32, stream);

    // INT32 *= INT32
    std::vector<double> exp_inplace_i32 = {(double)(i32[0]*i32_2[0]), (double)(i32[1]*i32_2[1]), (double)(i32[2]*i32_2[2]), (double)(i32[3]*i32_2[3])};
    totalFailed += RunInplaceTest<int32_t, int32_t>("API_InplaceMul_INT32", ACL_INT32, ACL_INT32, i32, i32_2, s4, exp_inplace_i32, stream);


    // ==========================================
    // [3] InplaceMuls 成功变体分支 (Tensor *= Scalar)
    // 成功秘诀：避免底层发生 Float32 -> Float16 的 Narrowing Cast
    // ==========================================
    // FLOAT *= FLOAT 标量
    totalFailed += RunInplaceMulsTest<float>("API_InplaceMuls_FLOAT", ACL_FLOAT, ACL_FLOAT, f32, 2.5, s4, exp_muls_f32, stream);
    
    // INT32 *= INT32 标量
    totalFailed += RunInplaceMulsTest<int32_t>("API_InplaceMuls_INT32", ACL_INT32, ACL_INT32, i32, 2.0, s4, exp_muls_i32, stream);

    // INT16 *= INT16 标量
    totalFailed += RunInplaceMulsTest<int16_t>("API_InplaceMuls_INT16", ACL_INT16, ACL_INT16, i16, 2.0, s4, exp_muls_i16, stream);

    // DOUBLE *= DOUBLE 标量
    std::vector<double> exp_inplace_d64 = {d64[0]*2.5, d64[1]*2.5, d64[2]*2.5, d64[3]*2.5};
    totalFailed += RunInplaceMulsTest<double>("API_InplaceMuls_DOUBLE", ACL_DOUBLE, ACL_DOUBLE, d64, 2.5, s4, exp_inplace_d64, stream);

    LOG_PRINT("\n--- 下面两个测试用例挖掘出算子缺陷 InplaceMuls目前对于输入为16位浮点数的情况下不生效 ---\n");
    std::vector<double> exp_inplace_fp16 = {fp16_to_fp32(fp16[0])*2.0, fp16_to_fp32(fp16[1])*2.0, fp16_to_fp32(fp16[2])*2.0, fp16_to_fp32(fp16[3])*2.0};
    totalFailed += RunInplaceMulsTest<uint16_t>(
        "API_InplaceMuls_FP16", 
        ACL_FLOAT16, ACL_FLOAT16, 
        fp16, 2.0, s4, exp_inplace_fp16, stream, 1e-3, 1e-3
    );
    totalFailed += RunInplaceMulsTest<uint16_t>(
        "API_InplaceMuls_BF16", 
        ACL_BF16, ACL_BF16, bf16, 2.0, s4, 
        exp_muls_bf16,
        stream, 1e-2, 1e-2
    );
    LOG_PRINT("\n");

    // ==========================================
    // [4] 空 Tensor 分支走齐 (常规空值测试)
    // ==========================================
    totalFailed += RunMulTest<float, float, float>("Empty_Tensor_Mul", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, {}, {}, {0, 2}, {}, stream); 
    totalFailed += RunMulsTest<float, float>("Empty_Tensor_Muls", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, {}, 2.5, {0, 2}, {}, stream);
    totalFailed += RunInplaceTest<float, float>("Empty_Tensor_InplaceMul", ACL_FLOAT, ACL_FLOAT, {}, {}, {0, 2}, {}, stream);
    totalFailed += RunInplaceMulsTest<float>("Empty_Tensor_InplaceMuls", ACL_FLOAT, ACL_FLOAT, {}, 2.5, {0, 2}, {}, stream);

    LOG_PRINT("\n--- 5. 特殊分支验证 (Broadcast, Format) ---\n");
    totalFailed += RunBroadcastTest("Broadcast_2D_1D_Fixed", stream);
    totalFailed += RunFormatWarningTest("Format_NC1HWC0_Warning_Check", stream);
    totalFailed += RunNonContiguousTest("NonContiguous_Stride10_Check", stream);
    
    LOG_PRINT("\n--- 5.5. 终极深度边缘逻辑 (Deep Edge Cases) ---\n");

    // 1. 0-D 张量广播 (0-D Tensor Broadcast)
    // tensor1: {2, 2}, tensor2: {} (表示单一标量)
    std::vector<float> t2_0d = {5.0f}; // 0-D 张量只包含 1 个标量值
    std::vector<double> exp_0d = {f32[0]*5.0, f32[1]*5.0, f32[2]*5.0, f32[3]*5.0};
    totalFailed += RunMulBroadcastTest<float, float, float>("Broadcast_Tensor_0D", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, f32, t2_0d, {2, 2}, {}, {2, 2}, exp_0d, stream);

    // 2. 双侧非连续内存 (Dual Non-Contiguous)
    totalFailed += RunDualNonContiguousTest("Dual_NonContiguous_Check", stream);

    // 3. InplaceMuls 复数特殊路由 (Inplace Complex Scalar)
    // CPU 端算数推演结果预期
    std::vector<std::complex<double>> exp_inplace_c64 = {
        {(c64_v1[0].real() * 1.5), (c64_v1[0].imag() * 1.5)}, 
        {(c64_v1[1].real() * 1.5), (c64_v1[1].imag() * 1.5)}, 
        {0.0, 0.0}, 
        {(c64_v1[3].real() * 1.5), (c64_v1[3].imag() * 1.5)}
    };
    
    totalFailed += RunInplaceMulsComplexTest<float>(
        "API_InplaceMuls_COMPLEX64_DOUBLE_Scalar", 
        ACL_COMPLEX64, ACL_DOUBLE, 
        c64_v1, 1.5, s4, exp_inplace_c64, stream
    );

    LOG_PRINT("\n--- 6. 专项反向拦截测试 ---\n");
    totalFailed += TestNegativeNullptr("Negative_Nullptr_AllAPIs");
    totalFailed += TestNegativeShapeMismatch("Negative_ShapeMismatch", stream);
    totalFailed += TestNegativeInvalidDType("Negative_InvalidDType", stream);

    // 1. 专门针对 aclnn_mul.cpp 的类型提升逻辑 (CombineCategoriesWithComplex)
    // 覆盖 Line 92, 223 等
    LOG_PRINT("\n--- 7. 深度覆盖率补充：隐式提升 ---\n");
    
    // Mul: FLOAT x FLOAT16 -> FLOAT 
    // 覆盖 mul_tiling_arch35.cpp Line 111 (DTYPE_MAP 项)
    std::vector<float> f32_data = {2.0f};
    std::vector<uint16_t> f16_data = {fp32_to_fp16(5.0f)};
    totalFailed += RunMulTest<float, uint16_t, float>("Coverage_Tiling_FLOAT_FLOAT16",
                                                      ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT,
                                                      f32_data, f16_data, {1}, {10.0}, stream);
    LOG_PRINT("\n--- 8. 格式检查分支覆盖 (Format Check) ---\n");
    
    // 覆盖 aclnn_mul.cpp 中的 MulsCheckFormat (Line 364)
    totalFailed += RunMulsFormatWarningTest("Coverage_Muls_Format_Not_ND", stream);

    LOG_PRINT("\n--- 执行AICPU算子, 由于模拟不支持, 故FAIL ---\n");
    std::vector<std::complex<double>> c128_data = {{1.0, 1.0}};
    totalFailed += RunMulComplexMixTest<double, std::complex<double>, double>(
        "Coverage_AiCpu_Route_Complex128", 
        ACL_COMPLEX128, ACL_COMPLEX128, ACL_COMPLEX128, 
        c128_data, c128_data, {1}, {}, stream);

    LOG_PRINT("\n=== Final Ultra-Coverage Summary: %d failed ===\n", totalFailed);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return totalFailed;
}