/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <complex>
#include <random>
#include <type_traits>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"
#include "aclnnop/aclnn_exp2.h"

#define CHECK_RET(cond, return_expr) \
  do { if (!(cond)) { return_expr; } } while (0)
#define LOG_PRINT(message, ...) \
  do { printf(message, ##__VA_ARGS__); } while (0)

// ============================================================================
// 1. Helper Utilities: Shape, Random & Precision Converters
// ============================================================================

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (auto i : shape) size *= i; 
    return size;
}

float GetRandFloat(float min = 0.1f, float max = 5.0f) {
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

int32_t GetRandInt(int32_t min = 1, int32_t max = 10) {
    static std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dis(min, max);
    return dis(gen);
}

// BFloat16 <-> Float32
uint16_t fp32_to_bf16(float f) {
    uint32_t res; std::memcpy(&res, &f, sizeof(f));
    return res >> 16;
}
float bf16_to_fp32(uint16_t b) {
    uint32_t res = b << 16;
    float f; std::memcpy(&f, &res, sizeof(f));
    return f;
}

// Float16 <-> Float32
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

aclScalar* CreateAclScalarFromDouble(double val, aclDataType dt) {
    switch (dt) {
        case ACL_FLOAT16: { uint16_t v = fp32_to_fp16(static_cast<float>(val)); return aclCreateScalar(&v, dt); }
        case ACL_BF16: { uint16_t v = fp32_to_bf16(static_cast<float>(val)); return aclCreateScalar(&v, dt); }
        case ACL_FLOAT: { float v = static_cast<float>(val); return aclCreateScalar(&v, dt); }
        case ACL_DOUBLE: { double v = val; return aclCreateScalar(&v, dt); }
        case ACL_INT8: { int8_t v = static_cast<int8_t>(val); return aclCreateScalar(&v, dt); }
        case ACL_UINT8: { uint8_t v = static_cast<uint8_t>(val); return aclCreateScalar(&v, dt); }
        case ACL_INT16: { int16_t v = static_cast<int16_t>(val); return aclCreateScalar(&v, dt); }
        case ACL_INT32: { int32_t v = static_cast<int32_t>(val); return aclCreateScalar(&v, dt); }
        case ACL_INT64: { int64_t v = static_cast<int64_t>(val); return aclCreateScalar(&v, dt); }
        case ACL_BOOL: { bool v = static_cast<bool>(val); return aclCreateScalar(&v, dt); }
        default: { float v = static_cast<float>(val); return aclCreateScalar(&v, dt); }
    }
}

aclScalar* CreateAclScalarFromComplex(std::complex<double> val, aclDataType dt) {
    if (dt == ACL_COMPLEX64) {
        float v[2] = {static_cast<float>(val.real()), static_cast<float>(val.imag())};
        return aclCreateScalar(v, dt);
    } else if (dt == ACL_COMPLEX128) {
        double v[2] = {val.real(), val.imag()};
        return aclCreateScalar(v, dt);
    }
    float v = static_cast<float>(val.real());
    return aclCreateScalar(&v, dt);
}

// ============================================================================
// 2. ACL Initialization & Tensor Creation
// ============================================================================

int Init(int32_t deviceId, aclrtStream* stream) {
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return 1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, return 1);
    CHECK_RET(aclrtCreateStream(stream) == ACL_SUCCESS, return 1);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor, aclFormat format = ACL_FORMAT_ND) {
    auto size = GetShapeSize(shape) * sizeof(T);
    if (size > 0) {
        aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    } else {
        *deviceAddr = nullptr; 
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--) strides[i] = shape[i+1] * strides[i+1];
    
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, 
                              strides.empty() ? nullptr : strides.data(), 0,
                              format, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// ============================================================================
// 3. Verification & Execution Runners
// ============================================================================

template <typename TOut>
int VerifyOutput(const char* name, void* outDev, std::vector<TOut>& outHost, int64_t n, 
                 const std::vector<double>& expected, double atol, double rtol, aclDataType dtOut) {
    if (n > 0) aclrtMemcpy(outHost.data(), n * sizeof(TOut), outDev, n * sizeof(TOut), ACL_MEMCPY_DEVICE_TO_HOST);
    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        double actual = 0;
        if constexpr (std::is_same<TOut, uint16_t>::value) {
            actual = (dtOut == ACL_BF16) ? bf16_to_fp32(outHost[i]) : fp16_to_fp32(outHost[i]);
        } else if constexpr (std::is_arithmetic<TOut>::value) {
            actual = static_cast<double>(outHost[i]);
        }
        if (!AlmostEqual(expected[i], actual, atol, rtol)) failed++;
    }
    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);
    return failed > 0 ? 1 : 0;
}

template <typename TOut>
int VerifyComplexOutput(const char* name, void* outDev, std::vector<std::complex<TOut>>& outHost, int64_t n, 
                        const std::vector<std::complex<double>>& expected, double atol, double rtol, aclDataType dtOut) {
    if (n > 0) aclrtMemcpy(outHost.data(), n * sizeof(std::complex<TOut>), outDev, n * sizeof(std::complex<TOut>), ACL_MEMCPY_DEVICE_TO_HOST);
    int failed = 0;
    for (int64_t i = 0; i < n; i++) {
        double act_r = static_cast<double>(outHost[i].real());
        double act_i = static_cast<double>(outHost[i].imag());
        if (!AlmostEqual(expected[i].real(), act_r, atol, rtol) || 
            !AlmostEqual(expected[i].imag(), act_i, atol, rtol)) {
            failed++;
        }
    }
    LOG_PRINT(failed == 0 ? "[PASS] %s\n" : "[FAIL] %s: %d mismatches\n", name, failed);
    return failed > 0 ? 1 : 0;
}

// ----------------------------------------------------------------------------
// Exp2
// ----------------------------------------------------------------------------
template <typename TIn, typename TOut>
int RunExp2Test(const char* name, aclDataType dtIn, aclDataType dtOut,
                const std::vector<TIn>& self, const std::vector<int64_t>& shape, 
                const std::vector<double>& expected, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *outDev=nullptr; aclTensor *selfT=nullptr, *outT=nullptr;
    CreateAclTensor(self, shape, &selfDev, dtIn, &selfT);
    std::vector<TOut> outHost(n > 0 ? n : 1, 0); 
    CreateAclTensor(outHost, shape, &outDev, dtOut, &outT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnExp2GetWorkspaceSize(selfT, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnExp2(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(selfT); aclDestroyTensor(outT);
    if(selfDev) aclrtFree(selfDev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

template <typename T>
int RunInplaceExp2Test(const char* name, aclDataType dt, const std::vector<T>& self,
                       const std::vector<int64_t>& shape, const std::vector<double>& expected, 
                       aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr; aclTensor *selfT=nullptr;
    std::vector<T> selfHostCopy = self;
    CreateAclTensor(selfHostCopy, shape, &selfDev, dt, &selfT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceExp2GetWorkspaceSize(selfT, &wsSize, &executor);

    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnInplaceExp2(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, selfDev, selfHostCopy, n, expected, atol, rtol, dt);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(selfT);
    if(selfDev) aclrtFree(selfDev);
    return fail_cnt;
}

// ----------------------------------------------------------------------------
// PowTensorTensor
// ----------------------------------------------------------------------------
template <typename T1, typename T2, typename TOut>
int RunPowTensorTensorTest(const char* name, aclDataType dt1, aclDataType dt2, aclDataType dtOut,
                           const std::vector<T1>& self, const std::vector<T2>& exp, 
                           const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2, const std::vector<int64_t>& outShape, 
                           const std::vector<double>& expected, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(outShape);
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
    
    CreateAclTensor(self, shape1, &x1Dev, dt1, &x1T);
    CreateAclTensor(exp, shape2, &x2Dev, dt2, &x2T);
    std::vector<TOut> outHost(n > 0 ? n : 1, 0);
    CreateAclTensor(outHost, outShape, &outDev, dtOut, &outT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorTensorGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnPowTensorTensor(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    if(x1Dev) aclrtFree(x1Dev); if(x2Dev) aclrtFree(x2Dev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

template <typename T1, typename T2>
int RunInplacePowTensorTensorTest(const char* name, aclDataType dt1, aclDataType dt2, 
                                  const std::vector<T1>& self, const std::vector<T2>& exp, 
                                  const std::vector<int64_t>& shape, const std::vector<double>& expected, 
                                  aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *expDev=nullptr; aclTensor *selfT=nullptr, *expT=nullptr;
    std::vector<T1> selfHostCopy = self;
    CreateAclTensor(selfHostCopy, shape, &selfDev, dt1, &selfT);
    CreateAclTensor(exp, shape, &expDev, dt2, &expT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplacePowTensorTensorGetWorkspaceSize(selfT, expT, &wsSize, &executor);

    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnInplacePowTensorTensor(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, selfDev, selfHostCopy, n, expected, atol, rtol, dt1);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(selfT); aclDestroyTensor(expT);
    if(selfDev) aclrtFree(selfDev); if(expDev) aclrtFree(expDev);
    return fail_cnt;
}

// ----------------------------------------------------------------------------
// PowTensorScalar
// ----------------------------------------------------------------------------
template <typename TIn, typename TOut>
int RunPowTensorScalarTest(const char* name, aclDataType dtIn, aclDataType dtScalar, aclDataType dtOut,
                           const std::vector<TIn>& self, double expVal, const std::vector<int64_t>& shape, 
                           const std::vector<double>& expected, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *outDev=nullptr; aclTensor *selfT=nullptr, *outT=nullptr;
    CreateAclTensor(self, shape, &selfDev, dtIn, &selfT);
    std::vector<TOut> outHost(n > 0 ? n : 1, 0); 
    CreateAclTensor(outHost, shape, &outDev, dtOut, &outT);

    aclScalar* expScalar = CreateAclScalarFromDouble(expVal, dtScalar);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorScalarGetWorkspaceSize(selfT, expScalar, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnPowTensorScalar(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(selfT); aclDestroyTensor(outT); aclDestroyScalar(expScalar);
    if(selfDev) aclrtFree(selfDev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

template <typename T>
int RunInplacePowTensorScalarTest(const char* name, aclDataType dt, aclDataType dtScalar, 
                                  const std::vector<T>& self, double expVal, const std::vector<int64_t>& shape, 
                                  const std::vector<double>& expected, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr; aclTensor *selfT=nullptr;
    std::vector<T> selfHostCopy = self;
    CreateAclTensor(selfHostCopy, shape, &selfDev, dt, &selfT);

    aclScalar* expScalar = CreateAclScalarFromDouble(expVal, dtScalar);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplacePowTensorScalarGetWorkspaceSize(selfT, expScalar, &wsSize, &executor);

    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnInplacePowTensorScalar(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, selfDev, selfHostCopy, n, expected, atol, rtol, dt);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(selfT); aclDestroyScalar(expScalar);
    if(selfDev) aclrtFree(selfDev);
    return fail_cnt;
}

// ----------------------------------------------------------------------------
// PowScalarTensor
// ----------------------------------------------------------------------------
template <typename TIn, typename TOut>
int RunPowScalarTensorTest(const char* name, aclDataType dtScalar, aclDataType dtIn, aclDataType dtOut,
                           double selfVal, const std::vector<TIn>& exp, const std::vector<int64_t>& shape, 
                           const std::vector<double>& expected, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *expDev=nullptr, *outDev=nullptr; aclTensor *expT=nullptr, *outT=nullptr;
    CreateAclTensor(exp, shape, &expDev, dtIn, &expT);
    std::vector<TOut> outHost(n > 0 ? n : 1, 0); 
    CreateAclTensor(outHost, shape, &outDev, dtOut, &outT);

    aclScalar* selfScalar = CreateAclScalarFromDouble(selfVal, dtScalar);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowScalarTensorGetWorkspaceSize(selfScalar, expT, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnPowScalarTensor(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(expT); aclDestroyTensor(outT); aclDestroyScalar(selfScalar);
    if(expDev) aclrtFree(expDev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

// ----------------------------------------------------------------------------
// Negative & Boundary Condition Tests
// ----------------------------------------------------------------------------
int TestNegativeNullptr(const char* name) {
    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    int fail = 0;
    if (aclnnPowTensorTensorGetWorkspaceSize(nullptr, nullptr, nullptr, &wsSize, &exec) == ACL_SUCCESS) fail++;
    if (aclnnPowTensorScalarGetWorkspaceSize(nullptr, nullptr, nullptr, &wsSize, &exec) == ACL_SUCCESS) fail++;
    if (aclnnPowScalarTensorGetWorkspaceSize(nullptr, nullptr, nullptr, &wsSize, &exec) == ACL_SUCCESS) fail++;
    if (aclnnExp2GetWorkspaceSize(nullptr, nullptr, &wsSize, &exec) == ACL_SUCCESS) fail++;
    LOG_PRINT(fail == 0 ? "[PASS] %s (Blocked)\n" : "[FAIL] %s (Did NOT block)\n", name);
    return fail;
}

int TestNegativeShapeMismatch(const char* name, aclrtStream stream) {
    int fail = 0;
    void *t1Dev=nullptr, *t2Dev=nullptr, *outDev=nullptr;
    aclTensor *t1=nullptr, *t2=nullptr, *outT=nullptr;

    CreateAclTensor<float>({1,2,3,4,5,6}, {2,3}, &t1Dev, ACL_FLOAT, &t1);
    CreateAclTensor<float>({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}, {4,5}, &t2Dev, ACL_FLOAT, &t2);
    CreateAclTensor<float>({0,0,0,0,0,0}, {2,3}, &outDev, ACL_FLOAT, &outT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    if (aclnnPowTensorTensorGetWorkspaceSize(t1, t2, outT, &wsSize, &executor) == ACL_SUCCESS) fail++;

    aclDestroyTensor(t1); aclDestroyTensor(t2); aclDestroyTensor(outT);
    aclrtFree(t1Dev); aclrtFree(t2Dev); aclrtFree(outDev);

    LOG_PRINT(fail == 0 ? "[PASS] %s (Blocked)\n" : "[FAIL] %s (Did NOT block)\n", name);
    return fail > 0 ? 1 : 0;
}

int TestNegativeInvalidDTypeAll(const char* name, aclrtStream stream) {
    int fail = 0;
    void *t1Dev=nullptr, *t2Dev=nullptr, *outDev=nullptr;
    aclTensor *t1=nullptr, *t2=nullptr, *outT=nullptr;

    // 注入非法的魔数 DType (9999) 作为测试依据
    CreateAclTensor<float>({1,2}, {2}, &t1Dev, static_cast<aclDataType>(9999), &t1);
    CreateAclTensor<float>({1,2}, {2}, &t2Dev, ACL_FLOAT, &t2);
    CreateAclTensor<float>({0,0}, {2}, &outDev, ACL_FLOAT, &outT);

    aclScalar* expScalar = CreateAclScalarFromDouble(2.0, static_cast<aclDataType>(9999));
    aclScalar* normalScalar = CreateAclScalarFromDouble(2.0, ACL_FLOAT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;

    // 1. Pow 家族 (Tensor-Tensor, Tensor-Scalar, Scalar-Tensor)
    if (aclnnPowTensorTensorGetWorkspaceSize(t1, t2, outT, &wsSize, &executor) == ACL_SUCCESS) fail++;
    if (aclnnPowTensorScalarGetWorkspaceSize(t1, normalScalar, outT, &wsSize, &executor) == ACL_SUCCESS) fail++;
    if (aclnnPowScalarTensorGetWorkspaceSize(expScalar, t2, outT, &wsSize, &executor) == ACL_SUCCESS) fail++;

    // 2. Pow 家族的 Inplace 变体
    if (aclnnInplacePowTensorTensorGetWorkspaceSize(t1, t2, &wsSize, &executor) == ACL_SUCCESS) fail++;
    if (aclnnInplacePowTensorScalarGetWorkspaceSize(t1, normalScalar, &wsSize, &executor) == ACL_SUCCESS) fail++;

    // 3. Exp2 家族
    if (aclnnExp2GetWorkspaceSize(t1, outT, &wsSize, &executor) == ACL_SUCCESS) fail++;
    if (aclnnInplaceExp2GetWorkspaceSize(t1, &wsSize, &executor) == ACL_SUCCESS) fail++;

    // 释放资源
    aclDestroyTensor(t1); aclDestroyTensor(t2); aclDestroyTensor(outT);
    aclDestroyScalar(expScalar); aclDestroyScalar(normalScalar);
    aclrtFree(t1Dev); aclrtFree(t2Dev); aclrtFree(outDev);

    LOG_PRINT(fail == 0 ? "[PASS] %s (Blocked all 7 APIs)\n" : "[FAIL] %s (Did NOT block %d times)\n", name, fail);
    return fail > 0 ? 1 : 0;
}

int TestFormatNZWarning(const char* name, aclrtStream stream) {
    int fail = 0;
    void *t1Dev=nullptr, *t2Dev=nullptr, *outDev=nullptr;
    aclTensor *t1=nullptr, *t2=nullptr, *outT=nullptr;

    CreateAclTensor<float>({1,2,3,4}, {4}, &t1Dev, ACL_FLOAT, &t1, ACL_FORMAT_FRACTAL_NZ);
    CreateAclTensor<float>({1,2,3,4}, {4}, &t2Dev, ACL_FLOAT, &t2, ACL_FORMAT_FRACTAL_NZ);
    CreateAclTensor<float>({0,0,0,0}, {4}, &outDev, ACL_FLOAT, &outT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    if (aclnnPowTensorTensorGetWorkspaceSize(t1, t2, outT, &wsSize, &executor) != ACL_SUCCESS) fail++;

    aclDestroyTensor(t1); aclDestroyTensor(t2); aclDestroyTensor(outT);
    aclrtFree(t1Dev); aclrtFree(t2Dev); aclrtFree(outDev);

    // Warning is logged but doesn't block execution, expecting SUCCESS
    LOG_PRINT(fail == 0 ? "[PASS] %s (Executed with warning)\n" : "[FAIL] %s (Failed)\n", name);
    return fail;
}

int TestNegativeExponentOnInt(const char* name, aclrtStream stream) {
    int fail = 0;
    void *t1Dev=nullptr, *outDev=nullptr;
    aclTensor *t1=nullptr, *outT=nullptr;

    CreateAclTensor<int32_t>({2,2,2,2}, {4}, &t1Dev, ACL_INT32, &t1);
    CreateAclTensor<int32_t>({0,0,0,0}, {4}, &outDev, ACL_INT32, &outT);
    aclScalar* expScalar = CreateAclScalarFromDouble(-1.0, ACL_INT32);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    // Base dtype integral and exponent negative -> should fail validation CheckPowTensorScalarExponet
    if (aclnnPowTensorScalarGetWorkspaceSize(t1, expScalar, outT, &wsSize, &executor) == ACL_SUCCESS) fail++;

    aclDestroyTensor(t1); aclDestroyTensor(outT); aclDestroyScalar(expScalar);
    aclrtFree(t1Dev); aclrtFree(outDev);

    LOG_PRINT(fail == 0 ? "[PASS] %s (Blocked)\n" : "[FAIL] %s (Did NOT block)\n", name);
    return fail;
}

int TestExponentOverflow(const char* name, aclrtStream stream) {
    int fail = 0;
    void *t1Dev=nullptr, *outDev=nullptr;
    aclTensor *t1=nullptr, *outT=nullptr;

    CreateAclTensor<int8_t>({2,2,2,2}, {4}, &t1Dev, ACL_INT8, &t1);
    CreateAclTensor<int8_t>({0,0,0,0}, {4}, &outDev, ACL_INT8, &outT);
    aclScalar* expScalar = CreateAclScalarFromDouble(300.0, ACL_INT64); // > 127, overflows INT8 promote type

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    // exponent overflows promote type -> should fail CheckNotOverflow
    if (aclnnPowTensorScalarGetWorkspaceSize(t1, expScalar, outT, &wsSize, &executor) == ACL_SUCCESS) fail++;

    aclDestroyTensor(t1); aclDestroyTensor(outT); aclDestroyScalar(expScalar);
    aclrtFree(t1Dev); aclrtFree(outDev);

    LOG_PRINT(fail == 0 ? "[PASS] %s (Blocked)\n" : "[FAIL] %s (Did NOT block)\n", name);
    return fail;
}

template <typename TOut>
int RunPowTensorTensorComplexTest(const char* name, aclDataType dt1, aclDataType dt2, aclDataType dtOut,
                                  const std::vector<std::complex<TOut>>& self, const std::vector<std::complex<TOut>>& exp, 
                                  const std::vector<int64_t>& shape, const std::vector<std::complex<double>>& expected, 
                                  aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
    
    CreateAclTensor(self, shape, &x1Dev, dt1, &x1T);
    CreateAclTensor(exp, shape, &x2Dev, dt2, &x2T);
    std::vector<std::complex<TOut>> outHost(n > 0 ? n : 1, {0.0, 0.0});
    CreateAclTensor(outHost, shape, &outDev, dtOut, &outT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowTensorTensorGetWorkspaceSize(x1T, x2T, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnPowTensorTensor(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyComplexOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT);
    if(x1Dev) aclrtFree(x1Dev); if(x2Dev) aclrtFree(x2Dev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

template <typename T>
int RunInplacePowTensorTensorComplexTest(const char* name, aclDataType dt1, aclDataType dt2, 
                                         const std::vector<std::complex<T>>& self, const std::vector<std::complex<T>>& exp, 
                                         const std::vector<int64_t>& shape, const std::vector<std::complex<double>>& expected, 
                                         aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *expDev=nullptr; aclTensor *selfT=nullptr, *expT=nullptr;
    std::vector<std::complex<T>> selfHostCopy = self;
    CreateAclTensor(selfHostCopy, shape, &selfDev, dt1, &selfT);
    CreateAclTensor(exp, shape, &expDev, dt2, &expT);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplacePowTensorTensorGetWorkspaceSize(selfT, expT, &wsSize, &executor);

    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnInplacePowTensorTensor(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyComplexOutput(name, selfDev, selfHostCopy, n, expected, atol, rtol, dt1);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(selfT); aclDestroyTensor(expT);
    if(selfDev) aclrtFree(selfDev); if(expDev) aclrtFree(expDev);
    return fail_cnt;
}

template <typename TOut>
int RunPowScalarTensorComplexTest(const char* name, aclDataType dtScalar, aclDataType dtIn, aclDataType dtOut,
                                  std::complex<double> selfVal, const std::vector<std::complex<TOut>>& exp, 
                                  const std::vector<int64_t>& shape, const std::vector<std::complex<double>>& expected, 
                                  aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *expDev=nullptr, *outDev=nullptr; aclTensor *expT=nullptr, *outT=nullptr;
    CreateAclTensor(exp, shape, &expDev, dtIn, &expT);
    std::vector<std::complex<TOut>> outHost(n > 0 ? n : 1, {0.0, 0.0}); 
    CreateAclTensor(outHost, shape, &outDev, dtOut, &outT);

    aclScalar* selfScalar = CreateAclScalarFromComplex(selfVal, dtScalar);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnPowScalarTensorGetWorkspaceSize(selfScalar, expT, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnPowScalarTensor(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyComplexOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(expT); aclDestroyTensor(outT); aclDestroyScalar(selfScalar);
    if(expDev) aclrtFree(expDev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

// ============================================================================
// MAIN Execution
// ============================================================================

int main() {
    int32_t deviceId = 0; aclrtStream stream;
    if (Init(deviceId, &stream) != 0) return 1;
    int totalFailed = 0;

    std::vector<int64_t> s4 = {4};
    
    std::vector<float> f32 = {1.5f, 2.0f, 3.5f, 4.0f};
    std::vector<float> f32_2 = {2.0f, 0.5f, 1.0f, 0.0f};
    std::vector<int32_t> i32 = {2, 3, 4, 5};
    std::vector<int32_t> i32_2 = {2, 3, 1, 0};
    std::vector<int8_t> i8 = {2, 3, 4, 5};
    std::vector<uint8_t> ui8 = {2, 3, 4, 5};
    std::vector<int16_t> i16 = {2, 3, 4, 5};
    std::vector<uint8_t> b8 = {1, 0, 1, 0}; 

    std::vector<uint16_t> fp16 = {fp32_to_fp16(f32[0]), fp32_to_fp16(f32[1]), fp32_to_fp16(f32[2]), fp32_to_fp16(f32[3])};
    std::vector<uint16_t> fp16_2 = {fp32_to_fp16(f32_2[0]), fp32_to_fp16(f32_2[1]), fp32_to_fp16(f32_2[2]), fp32_to_fp16(f32_2[3])};
    std::vector<uint16_t> bf16 = {fp32_to_bf16(f32[0]), fp32_to_bf16(f32[1]), fp32_to_bf16(f32[2]), fp32_to_bf16(f32[3])};
    std::vector<uint16_t> bf16_2 = {fp32_to_bf16(f32_2[0]), fp32_to_bf16(f32_2[1]), fp32_to_bf16(f32_2[2]), fp32_to_bf16(f32_2[3])};

    LOG_PRINT("\n--- 1. Exp2 API Tests ---\n");

    // ========================================================================
    // [1] 常规 Exp2 接口测试 (aclnnExp2)
    // 覆盖: FLOAT, FP16, BF16, DOUBLE, INT32, INT8, UINT8, INT16, BOOL
    // ========================================================================
    
    // 1.1 浮点类型 (FLOAT, FP16, BF16, DOUBLE)
    std::vector<double> exp_exp2_f32 = {std::pow(2.0, f32[0]), std::pow(2.0, f32[1]), std::pow(2.0, f32[2]), std::pow(2.0, f32[3])};
    totalFailed += RunExp2Test<float, float>("Exp2_FLOAT", ACL_FLOAT, ACL_FLOAT, f32, s4, exp_exp2_f32, stream);
    
    std::vector<double> exp_exp2_fp16 = {std::pow(2.0, fp16_to_fp32(fp16[0])), std::pow(2.0, fp16_to_fp32(fp16[1])), std::pow(2.0, fp16_to_fp32(fp16[2])), std::pow(2.0, fp16_to_fp32(fp16[3]))};
    totalFailed += RunExp2Test<uint16_t, uint16_t>("Exp2_FP16", ACL_FLOAT16, ACL_FLOAT16, fp16, s4, exp_exp2_fp16, stream, 1e-3, 1e-3);
    
    LOG_PRINT("\n=== 算子缺陷: Exp2不支持bf16类型 ===\n");
    std::vector<double> exp_exp2_bf16 = {std::pow(2.0, bf16_to_fp32(bf16[0])), std::pow(2.0, bf16_to_fp32(bf16[1])), std::pow(2.0, bf16_to_fp32(bf16[2])), std::pow(2.0, bf16_to_fp32(bf16[3]))};
    totalFailed += RunExp2Test<uint16_t, uint16_t>("Exp2_BF16", ACL_BF16, ACL_BF16, bf16, s4, exp_exp2_bf16, stream, 1e-2, 1e-2);

    // 构造 DOUBLE 类型测试数据
    LOG_PRINT("\n=== 算子缺陷: Exp2不支持double类型 ===\n");
    std::vector<double> f64 = {(double)f32[0], (double)f32[1], (double)f32[2], (double)f32[3]};
    std::vector<double> exp_exp2_f64 = {std::pow(2.0, f64[0]), std::pow(2.0, f64[1]), std::pow(2.0, f64[2]), std::pow(2.0, f64[3])};
    totalFailed += RunExp2Test<double, double>("Exp2_DOUBLE", ACL_DOUBLE, ACL_DOUBLE, f64, s4, exp_exp2_f64, stream);

    // 1.2 整型与布尔型 (源码中整型会被隐式Cast为Float，此处输出也定为Float)
    LOG_PRINT("\n=== 算子缺陷: exp2不支持整型输入并且浮点输出的情况 ===\n");
    std::vector<double> exp_exp2_i32 = {std::pow(2.0, i32[0]), std::pow(2.0, i32[1]), std::pow(2.0, i32[2]), std::pow(2.0, i32[3])};
    totalFailed += RunExp2Test<int32_t, float>("Exp2_INT32_to_FLOAT", ACL_INT32, ACL_FLOAT, i32, s4, exp_exp2_i32, stream);

    std::vector<double> exp_exp2_i8 = {std::pow(2.0, i8[0]), std::pow(2.0, i8[1]), std::pow(2.0, i8[2]), std::pow(2.0, i8[3])};
    totalFailed += RunExp2Test<int8_t, float>("Exp2_INT8_to_FLOAT", ACL_INT8, ACL_FLOAT, i8, s4, exp_exp2_i8, stream);

    std::vector<double> exp_exp2_ui8 = {std::pow(2.0, ui8[0]), std::pow(2.0, ui8[1]), std::pow(2.0, ui8[2]), std::pow(2.0, ui8[3])};
    totalFailed += RunExp2Test<uint8_t, float>("Exp2_UINT8_to_FLOAT", ACL_UINT8, ACL_FLOAT, ui8, s4, exp_exp2_ui8, stream);

    std::vector<double> exp_exp2_i16 = {std::pow(2.0, i16[0]), std::pow(2.0, i16[1]), std::pow(2.0, i16[2]), std::pow(2.0, i16[3])};
    totalFailed += RunExp2Test<int16_t, float>("Exp2_INT16_to_FLOAT", ACL_INT16, ACL_FLOAT, i16, s4, exp_exp2_i16, stream);

    std::vector<double> exp_exp2_b8 = {std::pow(2.0, b8[0]), std::pow(2.0, b8[1]), std::pow(2.0, b8[2]), std::pow(2.0, b8[3])};
    totalFailed += RunExp2Test<uint8_t, float>("Exp2_BOOL_to_FLOAT", ACL_BOOL, ACL_FLOAT, b8, s4, exp_exp2_b8, stream);


    // ========================================================================
    // [2] 原地 Exp2 接口测试 (aclnnInplaceExp2)
    // 根据 INPLACE_DTYPE_SUPPORT_LIST，原地操作仅支持浮点类型
    // ========================================================================
    totalFailed += RunInplaceExp2Test<float>("InplaceExp2_FLOAT", ACL_FLOAT, f32, s4, exp_exp2_f32, stream);
    totalFailed += RunInplaceExp2Test<uint16_t>("InplaceExp2_FP16", ACL_FLOAT16, fp16, s4, exp_exp2_fp16, stream, 1e-3, 1e-3);
    LOG_PRINT("\n=== 算子缺陷: InplaceExp2不支持BF16类型 ===\n");
    totalFailed += RunInplaceExp2Test<uint16_t>("InplaceExp2_BF16", ACL_BF16, bf16, s4, exp_exp2_bf16, stream, 1e-2, 1e-2);
    LOG_PRINT("\n=== 算子缺陷: InplaceExp2不支持double类型 ===\n");
    totalFailed += RunInplaceExp2Test<double>("InplaceExp2_DOUBLE", ACL_DOUBLE, f64, s4, exp_exp2_f64, stream);
    
    LOG_PRINT("\n--- 2. PowTensorTensor API Tests (Full API Coverage) ---\n");
    // ========================================================================
    // [2.1] 基础 Tiling Keys 及核心数据类型 (FP32, FP16, BF16, INT32, INT16, INT8, UINT8)
    // ========================================================================
    std::vector<double> exp_ptt_f32 = {std::pow(f32[0], f32_2[0]), std::pow(f32[1], f32_2[1]), std::pow(f32[2], f32_2[2]), std::pow(f32[3], f32_2[3])};
    totalFailed += RunPowTensorTensorTest<float, float, float>("PTT_FLOAT", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, f32, f32_2, s4, s4, s4, exp_ptt_f32, stream);
    
    std::vector<double> exp_ptt_fp16 = {std::pow(fp16_to_fp32(fp16[0]), fp16_to_fp32(fp16_2[0])), std::pow(fp16_to_fp32(fp16[1]), fp16_to_fp32(fp16_2[1])), std::pow(fp16_to_fp32(fp16[2]), fp16_to_fp32(fp16_2[2])), std::pow(fp16_to_fp32(fp16[3]), fp16_to_fp32(fp16_2[3]))};
    totalFailed += RunPowTensorTensorTest<uint16_t, uint16_t, uint16_t>("PTT_FP16", ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, fp16, fp16_2, s4, s4, s4, exp_ptt_fp16, stream, 1e-3, 1e-3);
    
    std::vector<double> exp_ptt_bf16 = {std::pow(bf16_to_fp32(bf16[0]), bf16_to_fp32(bf16_2[0])), std::pow(bf16_to_fp32(bf16[1]), bf16_to_fp32(bf16_2[1])), std::pow(bf16_to_fp32(bf16[2]), bf16_to_fp32(bf16_2[2])), std::pow(bf16_to_fp32(bf16[3]), bf16_to_fp32(bf16_2[3]))};
    totalFailed += RunPowTensorTensorTest<uint16_t, uint16_t, uint16_t>("PTT_BF16", ACL_BF16, ACL_BF16, ACL_BF16, bf16, bf16_2, s4, s4, s4, exp_ptt_bf16, stream, 1e-2, 1e-2);
    
    std::vector<double> exp_ptt_i32 = {std::pow((double)i32[0], (double)i32_2[0]), std::pow((double)i32[1], (double)i32_2[1]), std::pow((double)i32[2], (double)i32_2[2]), std::pow((double)i32[3], (double)i32_2[3])};
    totalFailed += RunPowTensorTensorTest<int32_t, int32_t, int32_t>("PTT_INT32", ACL_INT32, ACL_INT32, ACL_INT32, i32, i32_2, s4, s4, s4, exp_ptt_i32, stream);
    
    std::vector<double> exp_ptt_ui8 = {std::pow((double)ui8[0], (double)ui8[0]), std::pow((double)ui8[1], (double)ui8[1]), std::pow((double)ui8[2], (double)ui8[2]), std::pow((double)ui8[3], (double)ui8[3])};
    totalFailed += RunPowTensorTensorTest<uint8_t, uint8_t, uint8_t>("PTT_UINT8", ACL_UINT8, ACL_UINT8, ACL_UINT8, ui8, ui8, s4, s4, s4, exp_ptt_ui8, stream);
    
    std::vector<double> exp_ptt_i8 = {std::pow((double)i8[0], (double)i8[0]), std::pow((double)i8[1], (double)i8[1]), std::pow((double)i8[2], (double)i8[2]), std::pow((double)i8[3], (double)i8[3])};
    totalFailed += RunPowTensorTensorTest<int8_t, int8_t, int8_t>("PTT_INT8", ACL_INT8, ACL_INT8, ACL_INT8, i8, i8, s4, s4, s4, exp_ptt_i8, stream);
    
    std::vector<double> exp_ptt_i16 = {std::pow((double)i16[0], (double)i16[0]), std::pow((double)i16[1], (double)i16[1]), std::pow((double)i16[2], (double)i16[2]), std::pow((double)i16[3], (double)i16[3])};
    totalFailed += RunPowTensorTensorTest<int16_t, int16_t, int16_t>("PTT_INT16", ACL_INT16, ACL_INT16, ACL_INT16, i16, i16, s4, s4, s4, exp_ptt_i16, stream);

    // ========================================================================
    // [2.2] 补充 API 支持的其他数据类型 (DOUBLE, INT64, COMPLEX)
    // ========================================================================
    LOG_PRINT("\n=== 算子缺陷: PowTensorTensor不支持double类型 ===\n");
    std::vector<double> f64_2 = {(double)f32_2[0], (double)f32_2[1], (double)f32_2[2], (double)f32_2[3]};
    std::vector<double> exp_ptt_f64 = {std::pow(f64[0], f64_2[0]), std::pow(f64[1], f64_2[1]), std::pow(f64[2], f64_2[2]), std::pow(f64[3], f64_2[3])};
    totalFailed += RunPowTensorTensorTest<double, double, double>("PTT_DOUBLE", ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE, f64, f64_2, s4, s4, s4, exp_ptt_f64, stream);

    LOG_PRINT("\n=== 算子缺陷: PowTensorTensor不支持int64类型 ===\n");
    std::vector<int64_t> i64 = {2, 3, 4, 5};
    std::vector<int64_t> i64_2 = {2, 3, 1, 0};
    std::vector<double> exp_ptt_i64 = {std::pow((double)i64[0], (double)i64_2[0]), std::pow((double)i64[1], (double)i64_2[1]), std::pow((double)i64[2], (double)i64_2[2]), std::pow((double)i64[3], (double)i64_2[3])};
    totalFailed += RunPowTensorTensorTest<int64_t, int64_t, int64_t>("PTT_INT64", ACL_INT64, ACL_INT64, ACL_INT64, i64, i64_2, s4, s4, s4, exp_ptt_i64, stream);

    LOG_PRINT("\n=== 算子缺陷: PowTensorTensor不支持Complex64类型 ===\n");
    std::vector<std::complex<float>> c64_1 = {{1.0f, 2.0f}, {-3.0f, 4.0f}, {0.5f, -0.5f}, {2.0f, 0.0f}};
    std::vector<std::complex<float>> c64_2 = {{2.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f}, {0.0f, 2.0f}};
    std::vector<std::complex<double>> exp_ptt_c64 = {
        std::pow(std::complex<double>(c64_1[0].real(), c64_1[0].imag()), std::complex<double>(c64_2[0].real(), c64_2[0].imag())),
        std::pow(std::complex<double>(c64_1[1].real(), c64_1[1].imag()), std::complex<double>(c64_2[1].real(), c64_2[1].imag())),
        std::pow(std::complex<double>(c64_1[2].real(), c64_1[2].imag()), std::complex<double>(c64_2[2].real(), c64_2[2].imag())),
        std::pow(std::complex<double>(c64_1[3].real(), c64_1[3].imag()), std::complex<double>(c64_2[3].real(), c64_2[3].imag()))
    };
    totalFailed += RunPowTensorTensorComplexTest<float>("PTT_COMPLEX64", ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64, c64_1, c64_2, s4, exp_ptt_c64, stream);

    // ========================================================================
    // [2.3] 混合数据类型推导 (Mixed PromoteType)
    // ========================================================================
    // FLOAT ^ FP16 -> FLOAT
    LOG_PRINT("\n=== 算子缺陷: PowTensorTensor不支持混合数据类型情况 ===\n");
    std::vector<double> exp_ptt_mix_f32_fp16 = {std::pow(f32[0], fp16_to_fp32(fp16_2[0])), std::pow(f32[1], fp16_to_fp32(fp16_2[1])), std::pow(f32[2], fp16_to_fp32(fp16_2[2])), std::pow(f32[3], fp16_to_fp32(fp16_2[3]))};
    totalFailed += RunPowTensorTensorTest<float, uint16_t, float>("PTT_Mixed_FLOAT_FP16_to_FLOAT", ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, f32, fp16_2, s4, s4, s4, exp_ptt_mix_f32_fp16, stream, 1e-3, 1e-3);
    
    // FP16 ^ INT32 -> FP16
    std::vector<double> exp_ptt_mix_fp16_i32 = {std::pow(fp16_to_fp32(fp16[0]), (double)i32_2[0]), std::pow(fp16_to_fp32(fp16[1]), (double)i32_2[1]), std::pow(fp16_to_fp32(fp16[2]), (double)i32_2[2]), std::pow(fp16_to_fp32(fp16[3]), (double)i32_2[3])};
    totalFailed += RunPowTensorTensorTest<uint16_t, int32_t, uint16_t>("PTT_Mixed_FP16_INT32_to_FP16", ACL_FLOAT16, ACL_INT32, ACL_FLOAT16, fp16, i32_2, s4, s4, s4, exp_ptt_mix_fp16_i32, stream, 1e-3, 1e-3);

    // BOOL ^ INT32 -> INT32 (源码中拦死了 BOOL ^ BOOL，此处用合法混合推导走入正常分支)
    std::vector<double> exp_ptt_mix_bool_i32 = {std::pow((double)(b8[0]?1:0), (double)i32_2[0]), std::pow((double)(b8[1]?1:0), (double)i32_2[1]), std::pow((double)(b8[2]?1:0), (double)i32_2[2]), std::pow((double)(b8[3]?1:0), (double)i32_2[3])};
    totalFailed += RunPowTensorTensorTest<uint8_t, int32_t, int32_t>("PTT_Mixed_BOOL_INT32_to_INT32", ACL_BOOL, ACL_INT32, ACL_INT32, b8, i32_2, s4, s4, s4, exp_ptt_mix_bool_i32, stream);

    // ========================================================================
    // [2.4] 原地更新操作 (InplacePowTensorTensor)
    // 覆盖不同精度以及混合类型的安全更新 (self 的精度必须能容纳结果)
    // ========================================================================
    // 同类型 Inplace
    totalFailed += RunInplacePowTensorTensorTest<float, float>("InplacePTT_FLOAT", ACL_FLOAT, ACL_FLOAT, f32, f32_2, s4, exp_ptt_f32, stream);
    totalFailed += RunInplacePowTensorTensorTest<uint16_t, uint16_t>("InplacePTT_FP16", ACL_FLOAT16, ACL_FLOAT16, fp16, fp16_2, s4, exp_ptt_fp16, stream, 1e-3, 1e-3);
    totalFailed += RunInplacePowTensorTensorTest<uint16_t, uint16_t>("InplacePTT_BF16", ACL_BF16, ACL_BF16, bf16, bf16_2, s4, exp_ptt_bf16, stream, 1e-2, 1e-2);
    LOG_PRINT("\n=== 算子缺陷: InplacePowTensorTensor不支持double类型 ===\n");
    totalFailed += RunInplacePowTensorTensorTest<double, double>("InplacePTT_DOUBLE", ACL_DOUBLE, ACL_DOUBLE, f64, f64_2, s4, exp_ptt_f64, stream);
    totalFailed += RunInplacePowTensorTensorTest<int32_t, int32_t>("InplacePTT_INT32", ACL_INT32, ACL_INT32, i32, i32_2, s4, exp_ptt_i32, stream);
    LOG_PRINT("\n=== 算子缺陷: InplacePowTensorTensor不支持complex64类型 ===\n");
    totalFailed += RunInplacePowTensorTensorComplexTest<float>("InplacePTT_COMPLEX64", ACL_COMPLEX64, ACL_COMPLEX64, c64_1, c64_2, s4, exp_ptt_c64, stream);

    // 混合类型 Inplace (FLOAT ^= FP16)
    LOG_PRINT("\n=== 算子缺陷: InplacePowTensorTensor不支持混合数据类型 ===\n");
    totalFailed += RunInplacePowTensorTensorTest<float, uint16_t>("InplacePTT_Mixed_FLOAT_FP16", ACL_FLOAT, ACL_FLOAT16, f32, fp16_2, s4, exp_ptt_mix_f32_fp16, stream, 1e-3, 1e-3);

    LOG_PRINT("\n--- 3. PowTensorScalar API Tests & Special Exponents ---\n");
    // Normal Float ^ 2.5
    std::vector<double> exp_pts_norm = {std::pow(f32[0], 2.5), std::pow(f32[1], 2.5), std::pow(f32[2], 2.5), std::pow(f32[3], 2.5)};
    totalFailed += RunPowTensorScalarTest<float, float>("PTS_Normal_2.5", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, f32, 2.5, s4, exp_pts_norm, stream);

    // Special: 0.5 (Sqrt)
    std::vector<double> exp_pts_sqrt = {std::pow(f32[0], 0.5), std::pow(f32[1], 0.5), std::pow(f32[2], 0.5), std::pow(f32[3], 0.5)};
    totalFailed += RunPowTensorScalarTest<float, float>("PTS_Special_Sqrt_0.5", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, f32, 0.5, s4, exp_pts_sqrt, stream);

    // Special: 2.0 (Square)
    LOG_PRINT("\n=== 算子缺陷: PowTensorScalar不支持scalar为2的情况 ===\n");
    std::vector<double> exp_pts_sq = {std::pow(f32[0], 2.0), std::pow(f32[1], 2.0), std::pow(f32[2], 2.0), std::pow(f32[3], 2.0)};
    totalFailed += RunPowTensorScalarTest<float, float>("PTS_Special_Square_2.0", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, f32, 2.0, s4, exp_pts_sq, stream);

    // Special: 3.0 (Cube)
    std::vector<double> exp_pts_cube = {std::pow(f32[0], 3.0), std::pow(f32[1], 3.0), std::pow(f32[2], 3.0), std::pow(f32[3], 3.0)};
    totalFailed += RunPowTensorScalarTest<float, float>("PTS_Special_Cube_3.0", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, f32, 3.0, s4, exp_pts_cube, stream);

    // Special: -0.5
    std::vector<double> exp_pts_nsqrt = {std::pow(f32[0], -0.5), std::pow(f32[1], -0.5), std::pow(f32[2], -0.5), std::pow(f32[3], -0.5)};
    totalFailed += RunPowTensorScalarTest<float, float>("PTS_Special_NegSqrt_-0.5", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, f32, -0.5, s4, exp_pts_nsqrt, stream);

    // Special: -1.0
    std::vector<double> exp_pts_recip = {std::pow(f32[0], -1.0), std::pow(f32[1], -1.0), std::pow(f32[2], -1.0), std::pow(f32[3], -1.0)};
    totalFailed += RunPowTensorScalarTest<float, float>("PTS_Special_Reciprocal_-1.0", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, f32, -1.0, s4, exp_pts_recip, stream);

    // Special: -2.0
    std::vector<double> exp_pts_nsq = {std::pow(f32[0], -2.0), std::pow(f32[1], -2.0), std::pow(f32[2], -2.0), std::pow(f32[3], -2.0)};
    totalFailed += RunPowTensorScalarTest<float, float>("PTS_Special_NegSquare_-2.0", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, f32, -2.0, s4, exp_pts_nsq, stream);

    // Square cast list: Int8 ^ 2.0 -> Int32
    LOG_PRINT("\n=== 算子缺陷: PowTensorScalar不支持int8输入int32输出的情况 ===\n");
    std::vector<double> exp_pts_i8_sq = {std::pow((double)i8[0], 2.0), std::pow((double)i8[1], 2.0), std::pow((double)i8[2], 2.0), std::pow((double)i8[3], 2.0)};
    totalFailed += RunPowTensorScalarTest<int8_t, int32_t>("PTS_Square_Cast_INT8_to_INT32", ACL_INT8, ACL_INT64, ACL_INT32, i8, 2.0, s4, exp_pts_i8_sq, stream);
    
    // Inplace PTS
    totalFailed += RunInplacePowTensorScalarTest<float>("InplacePTS_FLOAT_2.5", ACL_FLOAT, ACL_FLOAT, f32, 2.5, s4, exp_pts_norm, stream);


    LOG_PRINT("\n--- 4. PowScalarTensor API Tests (Full API Coverage) ---\n");

    // ========================================================================
    // [4.1] 基础浮点类型 (FLOAT, FP16, BF16, DOUBLE)
    // ========================================================================
    std::vector<double> exp_pst_f32 = {std::pow(2.5, f32[0]), std::pow(2.5, f32[1]), std::pow(2.5, f32[2]), std::pow(2.5, f32[3])};
    totalFailed += RunPowScalarTensorTest<float, float>("PST_FLOAT", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 2.5, f32, s4, exp_pst_f32, stream);

    std::vector<double> exp_pst_fp16 = {std::pow(2.5, fp16_to_fp32(fp16[0])), std::pow(2.5, fp16_to_fp32(fp16[1])), std::pow(2.5, fp16_to_fp32(fp16[2])), std::pow(2.5, fp16_to_fp32(fp16[3]))};
    totalFailed += RunPowScalarTensorTest<uint16_t, uint16_t>("PST_FP16", ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, 2.5, fp16, s4, exp_pst_fp16, stream, 1e-3, 1e-3);

    std::vector<double> exp_pst_bf16 = {std::pow(2.5, bf16_to_fp32(bf16[0])), std::pow(2.5, bf16_to_fp32(bf16[1])), std::pow(2.5, bf16_to_fp32(bf16[2])), std::pow(2.5, bf16_to_fp32(bf16[3]))};
    totalFailed += RunPowScalarTensorTest<uint16_t, uint16_t>("PST_BF16", ACL_BF16, ACL_BF16, ACL_BF16, 2.5, bf16, s4, exp_pst_bf16, stream, 1e-2, 1e-2);

    LOG_PRINT("\n=== 算子缺陷: PowScalarTensor不支持double类型 ===\n");
    std::vector<double> exp_pst_f64 = {std::pow(2.5, f64[0]), std::pow(2.5, f64[1]), std::pow(2.5, f64[2]), std::pow(2.5, f64[3])};
    totalFailed += RunPowScalarTensorTest<double, double>("PST_DOUBLE", ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE, 2.5, f64, s4, exp_pst_f64, stream);

    // ========================================================================
    // [4.2] 基础整型 (INT32, INT64, INT8, UINT8, INT16)
    // ========================================================================
    std::vector<double> exp_pst_i32 = {std::pow(2.0, (double)i32[0]), std::pow(2.0, (double)i32[1]), std::pow(2.0, (double)i32[2]), std::pow(2.0, (double)i32[3])};
    totalFailed += RunPowScalarTensorTest<int32_t, int32_t>("PST_INT32", ACL_INT32, ACL_INT32, ACL_INT32, 2.0, i32, s4, exp_pst_i32, stream);

    LOG_PRINT("\n=== 算子缺陷: PowScalarTensor不支持int64类型 ===\n");
    std::vector<double> exp_pst_i64 = {std::pow(2.0, (double)i64[0]), std::pow(2.0, (double)i64[1]), std::pow(2.0, (double)i64[2]), std::pow(2.0, (double)i64[3])};
    totalFailed += RunPowScalarTensorTest<int64_t, int64_t>("PST_INT64", ACL_INT64, ACL_INT64, ACL_INT64, 2.0, i64, s4, exp_pst_i64, stream);

    std::vector<double> exp_pst_i8 = {std::pow(2.0, (double)i8[0]), std::pow(2.0, (double)i8[1]), std::pow(2.0, (double)i8[2]), std::pow(2.0, (double)i8[3])};
    totalFailed += RunPowScalarTensorTest<int8_t, int8_t>("PST_INT8", ACL_INT8, ACL_INT8, ACL_INT8, 2.0, i8, s4, exp_pst_i8, stream);

    std::vector<double> exp_pst_ui8 = {std::pow(2.0, (double)ui8[0]), std::pow(2.0, (double)ui8[1]), std::pow(2.0, (double)ui8[2]), std::pow(2.0, (double)ui8[3])};
    totalFailed += RunPowScalarTensorTest<uint8_t, uint8_t>("PST_UINT8", ACL_UINT8, ACL_UINT8, ACL_UINT8, 2.0, ui8, s4, exp_pst_ui8, stream);

    std::vector<double> exp_pst_i16 = {std::pow(2.0, (double)i16[0]), std::pow(2.0, (double)i16[1]), std::pow(2.0, (double)i16[2]), std::pow(2.0, (double)i16[3])};
    totalFailed += RunPowScalarTensorTest<int16_t, int16_t>("PST_INT16", ACL_INT16, ACL_INT16, ACL_INT16, 2.0, i16, s4, exp_pst_i16, stream);

    // ========================================================================
    // [4.3] 复数类型 (COMPLEX64)
    // ========================================================================
    LOG_PRINT("\n=== 算子缺陷: PowScalarTensor不支持Complex64类型 ===\n");
    std::complex<double> scalar_c64_val = {2.0, 1.0};
    std::vector<std::complex<double>> exp_pst_c64 = {
        std::pow(scalar_c64_val, std::complex<double>(c64_1[0].real(), c64_1[0].imag())),
        std::pow(scalar_c64_val, std::complex<double>(c64_1[1].real(), c64_1[1].imag())),
        std::pow(scalar_c64_val, std::complex<double>(c64_1[2].real(), c64_1[2].imag())),
        std::pow(scalar_c64_val, std::complex<double>(c64_1[3].real(), c64_1[3].imag()))
    };
    totalFailed += RunPowScalarTensorComplexTest<float>("PST_COMPLEX64", ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64, scalar_c64_val, c64_1, s4, exp_pst_c64, stream);

    // ========================================================================
    // [4.4] 混合数据类型推导 (Mixed PromoteType)
    // ========================================================================
    // Scalar INT32 ^ Tensor FLOAT -> FLOAT
    std::vector<double> exp_pst_mix_i32_f32 = {std::pow(3.0, f32[0]), std::pow(3.0, f32[1]), std::pow(3.0, f32[2]), std::pow(3.0, f32[3])};
    totalFailed += RunPowScalarTensorTest<float, float>("PST_Mixed_INT32_FLOAT_to_FLOAT", ACL_INT32, ACL_FLOAT, ACL_FLOAT, 3.0, f32, s4, exp_pst_mix_i32_f32, stream);

    // Scalar FLOAT ^ Tensor FP16 -> FLOAT
    LOG_PRINT("\n=== 算子缺陷: PowScalarTensor不支持混合数据类型 ===\n");
    std::vector<double> exp_pst_mix_f32_fp16 = {std::pow(2.5, fp16_to_fp32(fp16[0])), std::pow(2.5, fp16_to_fp32(fp16[1])), std::pow(2.5, fp16_to_fp32(fp16[2])), std::pow(2.5, fp16_to_fp32(fp16[3]))};
    totalFailed += RunPowScalarTensorTest<uint16_t, float>("PST_Mixed_FLOAT_FP16_to_FLOAT", ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, 2.5, fp16, s4, exp_pst_mix_f32_fp16, stream, 1e-3, 1e-3);

    // ========================================================================
    // [4.5] 特殊分支覆盖
    // ========================================================================
    // Fill(1) Branch: 1.0 ^ Float -> 触发底座为1时的特殊图构建分支
    LOG_PRINT("\n=== 算子缺陷: PowScalarTensor不支持scalar等于1的情况 ===\n");
    std::vector<double> exp_pst_fill1 = {1.0, 1.0, 1.0, 1.0};
    totalFailed += RunPowScalarTensorTest<float, float>("PST_Fill1_Branch", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1.0, f32, s4, exp_pst_fill1, stream);
    
    // Base 0.0 ^ Float -> 正常走 Kernel 处理，但结果特殊
    std::vector<double> exp_pst_base0 = {0.0, 0.0, 0.0, 0.0};
    totalFailed += RunPowScalarTensorTest<float, float>("PST_Base0_Branch", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0.0, f32, s4, exp_pst_base0, stream);

    LOG_PRINT("\n--- 5. Broadcast & Boundary & Empty Tensor Tests ---\n");
    // Broadcast
    std::vector<float> brd_x1 = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0}; 
    std::vector<float> brd_x2 = {2.0, 3.0, 2.0};             
    std::vector<double> exp_brd = {std::pow(2.0, 2.0), std::pow(3.0, 3.0), std::pow(4.0, 2.0), std::pow(5.0, 2.0), std::pow(6.0, 3.0), std::pow(7.0, 2.0)};
    totalFailed += RunPowTensorTensorTest<float, float, float>("PTT_Broadcast_2D_1D", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, brd_x1, brd_x2, {2, 3}, {3}, {2, 3}, exp_brd, stream);

    // Large Tensor
    std::vector<float> large_x(10000, 2.0f);
    std::vector<double> large_exp(10000, std::pow(2.0, 2.0));
    totalFailed += RunPowTensorTensorTest<float, float, float>("PTT_LargeShape_10K", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, large_x, large_x, {10000}, {10000}, {10000}, large_exp, stream);

    // Empty Tensor
    totalFailed += RunPowTensorTensorTest<float, float, float>("PTT_Empty_Tensor", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, {}, {}, {0, 2}, {0, 2}, {0, 2}, {}, stream);

    // Base 0, Exp 0 -> 1.0
    std::vector<float> base_0 = {0.0f, 0.0f};
    std::vector<float> exp_0 = {0.0f, 0.0f};
    std::vector<double> exp_0_0 = {1.0, 1.0};
    totalFailed += RunPowTensorTensorTest<float, float, float>("PTT_Boundary_0_pow_0", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, base_0, exp_0, {2}, {2}, {2}, exp_0_0, stream);


    LOG_PRINT("\n--- 6. Negative & Exception Interception Tests ---\n");
    totalFailed += TestNegativeNullptr("Negative_Nullptr");
    totalFailed += TestNegativeShapeMismatch("Negative_ShapeMismatch_PTT", stream);
    totalFailed += TestNegativeInvalidDTypeAll("Negative_InvalidDType", stream);
    totalFailed += TestFormatNZWarning("Warning_FRACTAL_NZ", stream);
    totalFailed += TestNegativeExponentOnInt("Negative_IntBase_NegativeExp_PTS", stream);
    totalFailed += TestExponentOverflow("Negative_ExponentOverflow_PTS", stream);

    LOG_PRINT("\n=== Final Pow Coverage Summary: %d failed ===\n", totalFailed);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return totalFailed;
}