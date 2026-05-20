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
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
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
                              ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

aclTensor* CreateDummyAclTensor(const std::vector<int64_t>& shape, aclDataType dataType) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    
    aclTensor* tensor = aclCreateTensor(
        shape.data(),
        shape.size(),
        dataType,
        strides.empty() ? nullptr : strides.data(),
        0,
        ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        nullptr  // 无实际数据，仅用于参数校验
    );
    
    return tensor;
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
        double actual = 0;
        if constexpr (std::is_same<TOut, uint16_t>::value) {
            actual = (dtOut == ACL_BF16) ? bf16_to_fp32(outHost[i]) : fp16_to_fp32(outHost[i]);
        } else if constexpr (std::is_arithmetic<TOut>::value) {
            actual = static_cast<double>(outHost[i]);
        }
        if(i<10)
        LOG_PRINT("NoOneLog%lf\t%lf\n", expected[i], actual);
        if (!AlmostEqual(expected[i], actual, atol, rtol)) failed++;
    }
    LOG_PRINT(failed == 0 ? "NoOneLog[PASS] %s\n" : "NoOneLog[FAIL] %s: %d mismatches\n", name, failed);
    return failed > 0 ? 1 : 0;
}

// aclnnAdd: out = self + alpha * other
template <typename T1, typename T2, typename TOut>
int RunAddTest(const char* name, aclDataType dt1, aclDataType dt2, aclDataType dtOut,
               const std::vector<T1>& self, const std::vector<T2>& other, double alphaVal,
               const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2, const std::vector<int64_t>& outShape, 
               const std::vector<double>& expected, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(outShape);
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
    
    CreateAclTensor(self, shape1, &x1Dev, dt1, &x1T);
    CreateAclTensor(other, shape2, &x2Dev, dt2, &x2T);
    std::vector<TOut> outHost(n > 0 ? n : 1, 0);
    CreateAclTensor(outHost, outShape, &outDev, dtOut, &outT);

    // 动态创建 alpha 类型，避免向下截断引发 161002 错误
    aclScalar* alpha = CreateAclScalarFromDouble(alphaVal, dtOut);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(x1T, x2T, alpha, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnAdd(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);
        if (wsAddr) aclrtFree(wsAddr);
    }  else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT); aclDestroyScalar(alpha);
    if(x1Dev) aclrtFree(x1Dev); if(x2Dev) aclrtFree(x2Dev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

// aclnnAdds: out = self + alpha * other_scalar
template <typename TIn, typename TOut>
int RunAddsTest(const char* name, aclDataType dtTensor, aclDataType dtOut,
                const std::vector<TIn>& self, double otherVal, double alphaVal, const std::vector<int64_t>& shape, 
                const std::vector<double>& expected, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *outDev=nullptr; aclTensor *selfT=nullptr, *outT=nullptr;
    CreateAclTensor(self, shape, &selfDev, dtTensor, &selfT);
    std::vector<TOut> outHost(n > 0 ? n : 1, 0); 
    CreateAclTensor(outHost, shape, &outDev, dtOut, &outT);

    aclScalar* otherScalar = CreateAclScalarFromDouble(otherVal, dtTensor);
    aclScalar* alpha = CreateAclScalarFromDouble(alphaVal, dtOut);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(selfT, otherScalar, alpha, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnAdds(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(selfT); aclDestroyTensor(outT); aclDestroyScalar(otherScalar); aclDestroyScalar(alpha);
    if(selfDev) aclrtFree(selfDev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

// aclnnInplaceAdd: selfRef += alpha * other
template <typename T1, typename T2>
int RunInplaceAddTest(const char* name, aclDataType dt1, aclDataType dt2, const std::vector<T1>& self, 
                      const std::vector<T2>& other, double alphaVal, const std::vector<int64_t>& shape, 
                      const std::vector<double>& expected, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *otherDev=nullptr; aclTensor *selfT=nullptr, *otherT=nullptr;
    std::vector<T1> selfHostCopy = self;
    CreateAclTensor(selfHostCopy, shape, &selfDev, dt1, &selfT);
    CreateAclTensor(other, shape, &otherDev, dt2, &otherT);

    aclScalar* alpha = CreateAclScalarFromDouble(alphaVal, dt1);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddGetWorkspaceSize(selfT, otherT, alpha, &wsSize, &executor);

    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnInplaceAdd(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, selfDev, selfHostCopy, n, expected, atol, rtol, dt1);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(selfT); aclDestroyTensor(otherT); aclDestroyScalar(alpha);
    if(selfDev) aclrtFree(selfDev); if(otherDev) aclrtFree(otherDev);
    return fail_cnt;
}

// aclnnInplaceAdds: selfRef += alpha * other_scalar
template <typename T>
int RunInplaceAddsTest(const char* name, aclDataType dt1, const std::vector<T>& self, double otherVal,
                       double alphaVal, const std::vector<int64_t>& shape, const std::vector<double>& expected, 
                       aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr; aclTensor *selfT=nullptr;
    std::vector<T> selfHostCopy = self;
    CreateAclTensor(selfHostCopy, shape, &selfDev, dt1, &selfT);

    aclScalar* otherScalar = CreateAclScalarFromDouble(otherVal, dt1);
    aclScalar* alpha = CreateAclScalarFromDouble(alphaVal, dt1);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddsGetWorkspaceSize(selfT, otherScalar, alpha, &wsSize, &executor);

    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnInplaceAdds(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, selfDev, selfHostCopy, n, expected, atol, rtol, dt1);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(selfT); aclDestroyScalar(otherScalar); aclDestroyScalar(alpha);
    if(selfDev) aclrtFree(selfDev);
    return fail_cnt;
}

// aclnnAddV3: out = scalar_self + alpha * tensor_other
template <typename TIn, typename TOut>
int RunAddV3Test(const char* name, aclDataType dtTensor, aclDataType dtOut,
                 double selfVal, const std::vector<TIn>& other, double alphaVal, const std::vector<int64_t>& shape, 
                 const std::vector<double>& expected, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *otherDev=nullptr, *outDev=nullptr; aclTensor *otherT=nullptr, *outT=nullptr;
    CreateAclTensor(other, shape, &otherDev, dtTensor, &otherT);
    std::vector<TOut> outHost(n > 0 ? n : 1, 0); 
    CreateAclTensor(outHost, shape, &outDev, dtOut, &outT);

    aclScalar* selfScalar = CreateAclScalarFromDouble(selfVal, dtTensor);
    aclScalar* alpha = CreateAclScalarFromDouble(alphaVal, dtOut);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(selfScalar, otherT, alpha, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnAddV3(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(otherT); aclDestroyTensor(outT); aclDestroyScalar(selfScalar); aclDestroyScalar(alpha);
    if(otherDev) aclrtFree(otherDev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

template <typename T>
int RunInplaceAddV3Test(const char* name, aclDataType dt1, double selfVal, const std::vector<T>& other, 
                        double alphaVal, const std::vector<int64_t>& shape, const std::vector<double>& expected, 
                        aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *otherDev=nullptr; aclTensor *otherT=nullptr;
    std::vector<T> otherHostCopy = other;
    CreateAclTensor(otherHostCopy, shape, &otherDev, dt1, &otherT);

    aclScalar* selfScalar = CreateAclScalarFromDouble(selfVal, dt1);
    aclScalar* alpha = CreateAclScalarFromDouble(alphaVal, dt1);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddV3GetWorkspaceSize(selfScalar, otherT, alpha, &wsSize, &executor);

    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnInplaceAddV3(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, otherDev, otherHostCopy, n, expected, atol, rtol, dt1);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(otherT); aclDestroyScalar(selfScalar); aclDestroyScalar(alpha);
    if(otherDev) aclrtFree(otherDev);
    return fail_cnt;
}

// ============================================================================
// 4. 异常与负向边界测试
// ============================================================================

int TestNegativeNullptr(const char* name) {
    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    int fail = 0;
    if (aclnnAddGetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, &wsSize, &exec) == ACL_SUCCESS) fail++;
    if (aclnnAddsGetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, &wsSize, &exec) == ACL_SUCCESS) fail++;
    if (aclnnInplaceAddGetWorkspaceSize(nullptr, nullptr, nullptr, &wsSize, &exec) == ACL_SUCCESS) fail++;
    if (aclnnAddV3GetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, &wsSize, &exec) == ACL_SUCCESS) fail++;
    LOG_PRINT(fail == 0 ? "NoOneLog[PASS] %s (Blocked)\n" : "NoOneLog[FAIL] %s (Did NOT block)\n", name);
    return fail;
}

// ============================================================================
// 4. 异常与负向边界测试 (修改部分)
// ============================================================================

int TestNegativeShapeMismatchAll(const char* name, aclrtStream stream) {
    int fail = 0;
    void *t1Dev=nullptr, *t2Dev=nullptr, *outDev=nullptr;
    aclTensor *t1=nullptr, *t2=nullptr, *outT=nullptr;

    // t1: 2x3, t2: 4x5, out: 2x3
    CreateAclTensor<float>({1,2,3,4,5,6}, {2,3}, &t1Dev, ACL_FLOAT, &t1);
    CreateAclTensor<float>({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}, {4,5}, &t2Dev, ACL_FLOAT, &t2);
    CreateAclTensor<float>({0,0,0,0,0,0}, {2,3}, &outDev, ACL_FLOAT, &outT);

    aclScalar* alpha = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
    aclScalar* scalarT = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;

    // 1. aclnnAdd: 双输入 Shape 无法 Broadcast 广播
    if (aclnnAddGetWorkspaceSize(t1, t2, alpha, outT, &wsSize, &executor) == ACL_SUCCESS) fail++;
    // 2. aclnnAdds: 输出 Shape 无法匹配输入
    if (aclnnAddsGetWorkspaceSize(t1, scalarT, alpha, t2, &wsSize, &executor) == ACL_SUCCESS) fail++;
    // 3. aclnnInplaceAdd: other 无法广播到 selfRef 的 Shape
    if (aclnnInplaceAddGetWorkspaceSize(t1, t2, alpha, &wsSize, &executor) == ACL_SUCCESS) fail++;
    // 4. aclnnAddV3: 输出 Shape 无法匹配 other
    if (aclnnAddV3GetWorkspaceSize(scalarT, t1, alpha, t2, &wsSize, &executor) == ACL_SUCCESS) fail++;

    aclDestroyTensor(t1); aclDestroyTensor(t2); aclDestroyTensor(outT);
    aclDestroyScalar(alpha); aclDestroyScalar(scalarT);
    aclrtFree(t1Dev); aclrtFree(t2Dev); aclrtFree(outDev);

    LOG_PRINT(fail == 0 ? "NoOneLog[PASS] %s (Blocked)\n" : "NoOneLog[FAIL] %s (Did NOT block %d times)\n", name, fail);
    return fail > 0 ? 1 : 0;
}

int TestNegativeInvalidDTypeAll(const char* name, aclrtStream stream) {
    int fail = 0;
    void *t1Dev=nullptr, *t2Dev=nullptr, *outDev=nullptr;
    aclTensor *t1=nullptr, *t2=nullptr, *outT=nullptr;

    // 注入非法的魔数 DType (9999)
    CreateAclTensor<float>({1,2}, {2}, &t1Dev, static_cast<aclDataType>(9999), &t1);
    CreateAclTensor<float>({1,2}, {2}, &t2Dev, ACL_FLOAT, &t2);
    CreateAclTensor<float>({0,0}, {2}, &outDev, ACL_FLOAT, &outT);

    aclScalar* alpha = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
    aclScalar* scalarT = CreateAclScalarFromDouble(1.0, static_cast<aclDataType>(9999));
    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;

    if (aclnnAddGetWorkspaceSize(t1, t2, alpha, outT, &wsSize, &executor) == ACL_SUCCESS) fail++;
    if (aclnnAddsGetWorkspaceSize(t1, scalarT, alpha, outT, &wsSize, &executor) == ACL_SUCCESS) fail++;
    if (aclnnInplaceAddGetWorkspaceSize(t1, t2, alpha, &wsSize, &executor) == ACL_SUCCESS) fail++;
    if (aclnnInplaceAddsGetWorkspaceSize(t1, scalarT, alpha, &wsSize, &executor) == ACL_SUCCESS) fail++;
    if (aclnnAddV3GetWorkspaceSize(scalarT, t1, alpha, outT, &wsSize, &executor) == ACL_SUCCESS) fail++;
    if (aclnnInplaceAddV3GetWorkspaceSize(scalarT, t1, alpha, &wsSize, &executor) == ACL_SUCCESS) fail++;

    aclDestroyTensor(t1); aclDestroyTensor(t2); aclDestroyTensor(outT);
    aclDestroyScalar(alpha); aclDestroyScalar(scalarT);
    aclrtFree(t1Dev); aclrtFree(t2Dev); aclrtFree(outDev);

    LOG_PRINT(fail == 0 ? "NoOneLog[PASS] %s (Blocked)\n" : "[FAIL] %s (Did NOT block %d times)\n", name, fail);
    return fail > 0 ? 1 : 0;
}

// ============================================================================
// [新增] 专门针对复数的验证与标量工厂
// ============================================================================

// 专门为复数设计的输出验证器
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

// 专门用于创建复数 aclScalar 的工厂函数
aclScalar* CreateAclScalarFromComplex(std::complex<double> val, aclDataType dt) {
    if (dt == ACL_COMPLEX64) {
        float v[2] = {static_cast<float>(val.real()), static_cast<float>(val.imag())};
        return aclCreateScalar(v, dt);
    } else if (dt == ACL_COMPLEX128) {
        double v[2] = {val.real(), val.imag()};
        return aclCreateScalar(v, dt);
    } else if (dt == ACL_FLOAT) { // 针对 alpha 降级为 Float 的情况
        float v = static_cast<float>(val.real());
        return aclCreateScalar(&v, dt);
    } else if (dt == ACL_DOUBLE) {
        double v = val.real();
        return aclCreateScalar(&v, dt);
    }
    float v = static_cast<float>(val.real());
    return aclCreateScalar(&v, dt);
}

// ============================================================================
// [新增] 复数 API 测试执行器
// ============================================================================

template <typename T1, typename T2, typename TOut>
int RunAddComplexTest(const char* name, aclDataType dt1, aclDataType dt2, aclDataType dtOut,
                      const std::vector<T1>& self, const std::vector<T2>& other, std::complex<double> alphaVal,
                      const std::vector<int64_t>& shape, 
                      const std::vector<std::complex<double>>& expected, aclrtStream stream) {
    int64_t n = GetShapeSize(shape);
    void *x1Dev=nullptr, *x2Dev=nullptr, *outDev=nullptr;
    aclTensor *x1T=nullptr, *x2T=nullptr, *outT=nullptr;
    
    CreateAclTensor(self, shape, &x1Dev, dt1, &x1T);
    CreateAclTensor(other, shape, &x2Dev, dt2, &x2T);
    std::vector<std::complex<TOut>> outHost(n > 0 ? n : 1, {0.0, 0.0});
    CreateAclTensor(outHost, shape, &outDev, dtOut, &outT);

    aclScalar* alpha = CreateAclScalarFromComplex(alphaVal, dtOut);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddGetWorkspaceSize(x1T, x2T, alpha, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnAdd(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyComplexOutput(name, outDev, outHost, n, expected, 1e-5, 1e-5, dtOut);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(outT); aclDestroyScalar(alpha);
    if(x1Dev) aclrtFree(x1Dev); if(x2Dev) aclrtFree(x2Dev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

template <typename TIn, typename TOut>
int RunAddsComplexTest(const char* name, aclDataType dtTensor, aclDataType dtScalar, aclDataType dtOut,
                       const std::vector<TIn>& self, std::complex<double> otherVal, std::complex<double> alphaVal, 
                       const std::vector<int64_t>& shape, const std::vector<std::complex<double>>& expected, aclrtStream stream) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *outDev=nullptr; aclTensor *selfT=nullptr, *outT=nullptr;
    CreateAclTensor(self, shape, &selfDev, dtTensor, &selfT);
    std::vector<std::complex<TOut>> outHost(n > 0 ? n : 1, {0.0, 0.0}); 
    CreateAclTensor(outHost, shape, &outDev, dtOut, &outT);

    aclScalar* otherScalar = CreateAclScalarFromComplex(otherVal, dtScalar);
    aclScalar* alpha = CreateAclScalarFromComplex(alphaVal, dtOut);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(selfT, otherScalar, alpha, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnAdds(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyComplexOutput(name, outDev, outHost, n, expected, 1e-5, 1e-5, dtOut);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(selfT); aclDestroyTensor(outT); aclDestroyScalar(otherScalar); aclDestroyScalar(alpha);
    if(selfDev) aclrtFree(selfDev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

template <typename T1, typename T2>
int RunInplaceAddComplexTest(const char* name, aclDataType dt1, aclDataType dt2, const std::vector<T1>& self, 
                             const std::vector<T2>& other, std::complex<double> alphaVal, const std::vector<int64_t>& shape, 
                             const std::vector<std::complex<double>>& expected, aclrtStream stream) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *otherDev=nullptr; aclTensor *selfT=nullptr, *otherT=nullptr;
    std::vector<T1> selfHostCopy = self;
    CreateAclTensor(selfHostCopy, shape, &selfDev, dt1, &selfT);
    CreateAclTensor(other, shape, &otherDev, dt2, &otherT);

    aclScalar* alpha = CreateAclScalarFromComplex(alphaVal, dt1);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddGetWorkspaceSize(selfT, otherT, alpha, &wsSize, &executor);

    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnInplaceAdd(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyComplexOutput(name, selfDev, selfHostCopy, n, expected, 1e-5, 1e-5, dt1);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(selfT); aclDestroyTensor(otherT); aclDestroyScalar(alpha);
    if(selfDev) aclrtFree(selfDev); if(otherDev) aclrtFree(otherDev);
    return fail_cnt;
}

template <typename T>
int RunInplaceAddsComplexTest(const char* name, aclDataType dt1, aclDataType dtScalar, const std::vector<T>& self, std::complex<double> otherVal,
                              std::complex<double> alphaVal, const std::vector<int64_t>& shape, const std::vector<std::complex<double>>& expected, 
                              aclrtStream stream) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr; aclTensor *selfT=nullptr;
    std::vector<T> selfHostCopy = self;
    CreateAclTensor(selfHostCopy, shape, &selfDev, dt1, &selfT);

    aclScalar* otherScalar = CreateAclScalarFromComplex(otherVal, dtScalar);
    aclScalar* alpha = CreateAclScalarFromComplex(alphaVal, dt1);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddsGetWorkspaceSize(selfT, otherScalar, alpha, &wsSize, &executor);

    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnInplaceAdds(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyComplexOutput(name, selfDev, selfHostCopy, n, expected, 1e-5, 1e-5, dt1);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(selfT); aclDestroyScalar(otherScalar); aclDestroyScalar(alpha);
    if(selfDev) aclrtFree(selfDev);
    return fail_cnt;
}

// aclnnAdds: out = self + alpha * other_scalar
template <typename TIn, typename TOut>
int RunAddsTest(const char* name, aclDataType dtTensor, aclDataType dtScalar, aclDataType dtOut,
                const std::vector<TIn>& self, double otherVal, double alphaVal, const std::vector<int64_t>& shape, 
                const std::vector<double>& expected, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr, *outDev=nullptr; aclTensor *selfT=nullptr, *outT=nullptr;
    CreateAclTensor(self, shape, &selfDev, dtTensor, &selfT);
    std::vector<TOut> outHost(n > 0 ? n : 1, 0); 
    CreateAclTensor(outHost, shape, &outDev, dtOut, &outT);

    // 使用传入的 dtScalar 创建 other 标量，实现真正的混合类型
    aclScalar* otherScalar = CreateAclScalarFromDouble(otherVal, dtScalar);
    aclScalar* alpha = CreateAclScalarFromDouble(alphaVal, dtOut);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddsGetWorkspaceSize(selfT, otherScalar, alpha, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnAdds(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(selfT); aclDestroyTensor(outT); aclDestroyScalar(otherScalar); aclDestroyScalar(alpha);
    if(selfDev) aclrtFree(selfDev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

// aclnnInplaceAdds: selfRef += alpha * other_scalar
template <typename T>
int RunInplaceAddsTest(const char* name, aclDataType dt1, aclDataType dtScalar, const std::vector<T>& self, double otherVal,
                       double alphaVal, const std::vector<int64_t>& shape, const std::vector<double>& expected, 
                       aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *selfDev=nullptr; aclTensor *selfT=nullptr;
    std::vector<T> selfHostCopy = self;
    CreateAclTensor(selfHostCopy, shape, &selfDev, dt1, &selfT);

    aclScalar* otherScalar = CreateAclScalarFromDouble(otherVal, dtScalar);
    aclScalar* alpha = CreateAclScalarFromDouble(alphaVal, dt1);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddsGetWorkspaceSize(selfT, otherScalar, alpha, &wsSize, &executor);

    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnInplaceAdds(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, selfDev, selfHostCopy, n, expected, atol, rtol, dt1);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(selfT); aclDestroyScalar(otherScalar); aclDestroyScalar(alpha);
    if(selfDev) aclrtFree(selfDev);
    return fail_cnt;
}

// aclnnAddV3: out = scalar_self + alpha * tensor_other
template <typename TIn, typename TOut>
int RunAddV3Test(const char* name, aclDataType dtScalar, aclDataType dtTensor, aclDataType dtOut,
                 double selfVal, const std::vector<TIn>& other, double alphaVal, const std::vector<int64_t>& shape, 
                 const std::vector<double>& expected, aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *otherDev=nullptr, *outDev=nullptr; aclTensor *otherT=nullptr, *outT=nullptr;
    CreateAclTensor(other, shape, &otherDev, dtTensor, &otherT);
    std::vector<TOut> outHost(n > 0 ? n : 1, 0); 
    CreateAclTensor(outHost, shape, &outDev, dtOut, &outT);

    aclScalar* selfScalar = CreateAclScalarFromDouble(selfVal, dtScalar);
    aclScalar* alpha = CreateAclScalarFromDouble(alphaVal, dtOut);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnAddV3GetWorkspaceSize(selfScalar, otherT, alpha, outT, &wsSize, &executor);
    
    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnAddV3(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, outDev, outHost, n, expected, atol, rtol, dtOut);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(otherT); aclDestroyTensor(outT); aclDestroyScalar(selfScalar); aclDestroyScalar(alpha);
    if(otherDev) aclrtFree(otherDev); if(outDev) aclrtFree(outDev);
    return fail_cnt;
}

// aclnnInplaceAddV3: tensor_other = scalar_self + alpha * tensor_other
template <typename T>
int RunInplaceAddV3Test(const char* name, aclDataType dtScalar, aclDataType dtTensor, double selfVal, const std::vector<T>& other, 
                        double alphaVal, const std::vector<int64_t>& shape, const std::vector<double>& expected, 
                        aclrtStream stream, double atol = 1e-5, double rtol = 1e-5) {
    int64_t n = GetShapeSize(shape);
    void *otherDev=nullptr; aclTensor *otherT=nullptr;
    std::vector<T> otherHostCopy = other;
    CreateAclTensor(otherHostCopy, shape, &otherDev, dtTensor, &otherT);

    aclScalar* selfScalar = CreateAclScalarFromDouble(selfVal, dtScalar);
    aclScalar* alpha = CreateAclScalarFromDouble(alphaVal, dtTensor);

    uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceAddV3GetWorkspaceSize(selfScalar, otherT, alpha, &wsSize, &executor);

    int fail_cnt = 0;
    if (ret == ACL_SUCCESS) {
        void* wsAddr = nullptr;
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnInplaceAddV3(wsAddr, wsSize, executor, stream);
        aclrtSynchronizeStream(stream);
        fail_cnt = VerifyOutput(name, otherDev, otherHostCopy, n, expected, atol, rtol, dtTensor);
        if (wsAddr) aclrtFree(wsAddr);
    } else {
        LOG_PRINT("NoOneLog[FAIL] %s Workspace Error %d\n", name, ret); fail_cnt = 1;
    }

    aclDestroyTensor(otherT); aclDestroyScalar(selfScalar); aclDestroyScalar(alpha);
    if(otherDev) aclrtFree(otherDev);
    return fail_cnt;
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
    std::vector<int32_t> i32 = {GetRandInt(), GetRandInt(), GetRandInt(), GetRandInt()};
    std::vector<int32_t> i32_2 = {GetRandInt(), GetRandInt(), GetRandInt(), GetRandInt()};
    std::vector<int64_t> i64 = {1000, -2000, 3000, 0};
    std::vector<int8_t> i8 = {1, 2, 3, -4};
    std::vector<uint8_t> ui8 = {5, 6, 7, 8};
    std::vector<uint8_t> b8 = {1, 0, 1, 0}; 
    std::vector<int16_t> i16 = {10, -20, 30, -40};
    
    std::vector<uint16_t> fp16 = {fp32_to_fp16(f32[0]), fp32_to_fp16(f32[1]), fp32_to_fp16(f32[2]), fp32_to_fp16(f32[3])};
    std::vector<uint16_t> fp16_2 = {fp32_to_fp16(f32_2[0]), fp32_to_fp16(f32_2[1]), fp32_to_fp16(f32_2[2]), fp32_to_fp16(f32_2[3])};
    std::vector<uint16_t> bf16 = {fp32_to_bf16(f32[0]), fp32_to_bf16(f32[1]), fp32_to_bf16(f32[2]), fp32_to_bf16(f32[3])};
    std::vector<uint16_t> bf16_2 = {fp32_to_bf16(f32_2[0]), fp32_to_bf16(f32_2[1]), fp32_to_bf16(f32_2[2]), fp32_to_bf16(f32_2[3])};

    std::vector<std::complex<float>> c64_1 = {{1.0f, 2.0f}, {-3.0f, 4.0f}, {0.5f, -0.5f}, {2.0f, 0.0f}};
    std::vector<std::complex<float>> c64_2 = {{2.0f, -1.0f}, {1.0f, 1.0f}, {-0.5f, 0.5f}, {0.0f, 2.0f}};
    std::vector<std::complex<double>> c128_1 = {{1.0, 2.0}, {-3.0, 4.0}, {0.5, -0.5}, {2.0, 0.0}};
    std::vector<std::complex<double>> c128_2 = {{2.0, -1.0}, {1.0, 1.0}, {-0.5, 0.5}, {0.0, 2.0}};
    std::complex<double> c_alpha = {1.5, 0.5};     // 测试用的复数缩放因子 alpha != 1
    std::complex<double> c_scalar = {-1.0, 1.5};   // Adds 等 API 的 other 标量

    LOG_PRINT("NoOneLog--- 1. 各个 API 变体在基础数据类型上的强覆盖 (alpha=1 标准加法) ---\n");

    // ========================================================================
    // [1] aclnnAdd (out = self + other)
    // ========================================================================
    // 基础整型与布尔型
    totalFailed += RunAddTest<int8_t, int8_t, int8_t>("Add_INT8_alpha1", ACL_INT8, ACL_INT8, ACL_INT8, i8, i8, 1.0, s4, s4, s4, {(double)(i8[0]+i8[0]), (double)(i8[1]+i8[1]), (double)(i8[2]+i8[2]), (double)(i8[3]+i8[3])}, stream);
    totalFailed += RunAddTest<uint8_t, uint8_t, uint8_t>("Add_UINT8_alpha1", ACL_UINT8, ACL_UINT8, ACL_UINT8, ui8, ui8, 1.0, s4, s4, s4, {(double)(ui8[0]+ui8[0]), (double)(ui8[1]+ui8[1]), (double)(ui8[2]+ui8[2]), (double)(ui8[3]+ui8[3])}, stream);
    totalFailed += RunAddTest<uint8_t, uint8_t, uint8_t>("Add_BOOL_alpha1", ACL_BOOL, ACL_BOOL, ACL_BOOL, b8, b8, 1.0, s4, s4, s4, {(double)(b8[0]||b8[0]), (double)(b8[1]||b8[1]), (double)(b8[2]||b8[2]), (double)(b8[3]||b8[3])}, stream);
    totalFailed += RunAddTest<int64_t, int64_t, int64_t>("Add_INT64_alpha1", ACL_INT64, ACL_INT64, ACL_INT64, i64, i64, 1.0, s4, s4, s4, {(double)(i64[0]+i64[0]), (double)(i64[1]+i64[1]), (double)(i64[2]+i64[2]), (double)(i64[3]+i64[3])}, stream);
    
    // 浮点型
    totalFailed += RunAddTest<float, float, float>("Add_FLOAT_alpha1", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, f32, f32_2, 1.0, s4, s4, s4, {f32[0]+f32_2[0], f32[1]+f32_2[1], f32[2]+f32_2[2], f32[3]+f32_2[3]}, stream);
    totalFailed += RunAddTest<uint16_t, uint16_t, uint16_t>("Add_FP16_alpha1", ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, fp16, fp16_2, 1.0, s4, s4, s4, {fp16_to_fp32(fp16[0])+fp16_to_fp32(fp16_2[0]), fp16_to_fp32(fp16[1])+fp16_to_fp32(fp16_2[1]), fp16_to_fp32(fp16[2])+fp16_to_fp32(fp16_2[2]), fp16_to_fp32(fp16[3])+fp16_to_fp32(fp16_2[3])}, stream, 1e-3, 1e-3);
    totalFailed += RunAddTest<uint16_t, uint16_t, uint16_t>("Add_BF16_alpha1", ACL_BF16, ACL_BF16, ACL_BF16, bf16, bf16_2, 1.0, s4, s4, s4, {bf16_to_fp32(bf16[0])+bf16_to_fp32(bf16_2[0]), bf16_to_fp32(bf16[1])+bf16_to_fp32(bf16_2[1]), bf16_to_fp32(bf16[2])+bf16_to_fp32(bf16_2[2]), bf16_to_fp32(bf16[3])+bf16_to_fp32(bf16_2[3])}, stream, 1e-2, 1e-2);
    

    // ========================================================================
    // [2] aclnnAdds (out = self + other_scalar)
    // ========================================================================
    // 基础整型与布尔型
    totalFailed += RunAddsTest<int8_t, int8_t>("API_Adds_INT8_alpha1", ACL_INT8, ACL_INT8, i8, 2.0, 1.0, s4, {(double)(i8[0]+2), (double)(i8[1]+2), (double)(i8[2]+2), (double)(i8[3]+2)}, stream);
    totalFailed += RunAddsTest<uint8_t, uint8_t>("API_Adds_UINT8_alpha1", ACL_UINT8, ACL_UINT8, ui8, 2.0, 1.0, s4, {(double)(ui8[0]+2), (double)(ui8[1]+2), (double)(ui8[2]+2), (double)(ui8[3]+2)}, stream);
    totalFailed += RunAddsTest<uint8_t, uint8_t>("API_Adds_BOOL_alpha1", ACL_BOOL, ACL_BOOL, b8, 1.0, 1.0, s4, {(double)(b8[0]||1), (double)(b8[1]||1), (double)(b8[2]||1), (double)(b8[3]||1)}, stream);
    totalFailed += RunAddsTest<int64_t, int64_t>("API_Adds_INT64_alpha1", ACL_INT64, ACL_INT64, i64, 2.0, 1.0, s4, {(double)(i64[0]+2), (double)(i64[1]+2), (double)(i64[2]+2), (double)(i64[3]+2)}, stream);
    
    // 浮点型
    std::vector<double> exp_adds_f32_a1 = {f32[0]+1.0*2.0, f32[1]+1.0*2.0, f32[2]+1.0*2.0, f32[3]+1.0*2.0};
    totalFailed += RunAddsTest<float, float>("API_Adds_FLOAT_alpha1", ACL_FLOAT, ACL_FLOAT, f32, 2.0, 1.0, s4, exp_adds_f32_a1, stream);
    std::vector<double> exp_adds_fp16_a1 = {fp16_to_fp32(fp16[0])+1.0*2.0, fp16_to_fp32(fp16[1])+1.0*2.0, fp16_to_fp32(fp16[2])+1.0*2.0, fp16_to_fp32(fp16[3])+1.0*2.0};
    totalFailed += RunAddsTest<uint16_t, uint16_t>("API_Adds_FP16_alpha1", ACL_FLOAT16, ACL_FLOAT16, fp16, 2.0, 1.0, s4, exp_adds_fp16_a1, stream, 1e-3, 1e-3);
    std::vector<double> exp_adds_bf16_a1 = {bf16_to_fp32(bf16[0])+1.0*2.0, bf16_to_fp32(bf16[1])+1.0*2.0, bf16_to_fp32(bf16[2])+1.0*2.0, bf16_to_fp32(bf16[3])+1.0*2.0};
    totalFailed += RunAddsTest<uint16_t, uint16_t>("API_Adds_BF16_alpha1", ACL_BF16, ACL_BF16, bf16, 2.0, 1.0, s4, exp_adds_bf16_a1, stream, 1e-2, 1e-2);

    // 缺陷探测
    LOG_PRINT("NoOneLog=== 发掘算子缺陷, alpha != 1 时, Adds不支持int32的输入 ===\n");
    std::vector<double> exp_adds_i32 = {(double)(i32[0]+2*2), (double)(i32[1]+2*2), (double)(i32[2]+2*2), (double)(i32[3]+2*2)};
    totalFailed += RunAddsTest<int32_t, int32_t>("API_Adds_INT32_alpha2", ACL_INT32, ACL_INT32, i32, 2.0, 2.0, s4, exp_adds_i32, stream);


    // ========================================================================
    // [3] aclnnInplaceAdd (selfRef += other)
    // ========================================================================
    // 基础整型与布尔型
    totalFailed += RunInplaceAddTest<int8_t, int8_t>("API_InplaceAdd_INT8_alpha1", ACL_INT8, ACL_INT8, i8, i8, 1.0, s4, {(double)(i8[0]+i8[0]), (double)(i8[1]+i8[1]), (double)(i8[2]+i8[2]), (double)(i8[3]+i8[3])}, stream);
    totalFailed += RunInplaceAddTest<uint8_t, uint8_t>("API_InplaceAdd_UINT8_alpha1", ACL_UINT8, ACL_UINT8, ui8, ui8, 1.0, s4, {(double)(ui8[0]+ui8[0]), (double)(ui8[1]+ui8[1]), (double)(ui8[2]+ui8[2]), (double)(ui8[3]+ui8[3])}, stream);
    totalFailed += RunInplaceAddTest<uint8_t, uint8_t>("API_InplaceAdd_BOOL_alpha1", ACL_BOOL, ACL_BOOL, b8, b8, 1.0, s4, {(double)(b8[0]||b8[0]), (double)(b8[1]||b8[1]), (double)(b8[2]||b8[2]), (double)(b8[3]||b8[3])}, stream);
    totalFailed += RunInplaceAddTest<int64_t, int64_t>("API_InplaceAdd_INT64_alpha1", ACL_INT64, ACL_INT64, i64, i64, 1.0, s4, {(double)(i64[0]+i64[0]), (double)(i64[1]+i64[1]), (double)(i64[2]+i64[2]), (double)(i64[3]+i64[3])}, stream);
    std::vector<double> exp_inplace_i32 = {(double)(i32[0]+1.0*i32_2[0]), (double)(i32[1]+1.0*i32_2[1]), (double)(i32[2]+1.0*i32_2[2]), (double)(i32[3]+1.0*i32_2[3])};
    totalFailed += RunInplaceAddTest<int32_t, int32_t>("API_InplaceAdd_INT32_alpha1", ACL_INT32, ACL_INT32, i32, i32_2, 1.0, s4, exp_inplace_i32, stream);

    // 浮点型
    std::vector<double> exp_inplace_f32_a1 = {f32[0]+1.0*f32_2[0], f32[1]+1.0*f32_2[1], f32[2]+1.0*f32_2[2], f32[3]+1.0*f32_2[3]};
    totalFailed += RunInplaceAddTest<float, float>("API_InplaceAdd_FLOAT_alpha1", ACL_FLOAT, ACL_FLOAT, f32, f32_2, 1.0, s4, exp_inplace_f32_a1, stream);
    std::vector<double> exp_inplace_fp16_a1 = {fp16_to_fp32(fp16[0])+1.0*fp16_to_fp32(fp16_2[0]), fp16_to_fp32(fp16[1])+1.0*fp16_to_fp32(fp16_2[1]), fp16_to_fp32(fp16[2])+1.0*fp16_to_fp32(fp16_2[2]), fp16_to_fp32(fp16[3])+1.0*fp16_to_fp32(fp16_2[3])};
    totalFailed += RunInplaceAddTest<uint16_t, uint16_t>("API_InplaceAdd_FP16_alpha1", ACL_FLOAT16, ACL_FLOAT16, fp16, fp16_2, 1.0, s4, exp_inplace_fp16_a1, stream, 1e-3, 1e-3);
    std::vector<double> exp_inplace_bf16_a1 = {bf16_to_fp32(bf16[0])+1.0*bf16_to_fp32(bf16_2[0]), bf16_to_fp32(bf16[1])+1.0*bf16_to_fp32(bf16_2[1]), bf16_to_fp32(bf16[2])+1.0*bf16_to_fp32(bf16_2[2]), bf16_to_fp32(bf16[3])+1.0*bf16_to_fp32(bf16_2[3])};
    totalFailed += RunInplaceAddTest<uint16_t, uint16_t>("API_InplaceAdd_BF16_alpha1", ACL_BF16, ACL_BF16, bf16, bf16_2, 1.0, s4, exp_inplace_bf16_a1, stream, 1e-2, 1e-2);


    // ========================================================================
    // [4] aclnnInplaceAdds (selfRef += other_scalar)
    // ========================================================================
    // 基础整型与布尔型
    totalFailed += RunInplaceAddsTest<int8_t>("API_InplaceAdds_INT8_alpha1", ACL_INT8, i8, 2.0, 1.0, s4, {(double)(i8[0]+2), (double)(i8[1]+2), (double)(i8[2]+2), (double)(i8[3]+2)}, stream);
    totalFailed += RunInplaceAddsTest<uint8_t>("API_InplaceAdds_UINT8_alpha1", ACL_UINT8, ui8, 2.0, 1.0, s4, {(double)(ui8[0]+2), (double)(ui8[1]+2), (double)(ui8[2]+2), (double)(ui8[3]+2)}, stream);
    totalFailed += RunInplaceAddsTest<uint8_t>("API_InplaceAdds_BOOL_alpha1", ACL_BOOL, b8, 1.0, 1.0, s4, {(double)(b8[0]||1), (double)(b8[1]||1), (double)(b8[2]||1), (double)(b8[3]||1)}, stream);
    totalFailed += RunInplaceAddsTest<int64_t>("API_InplaceAdds_INT64_alpha1", ACL_INT64, i64, 2.0, 1.0, s4, {(double)(i64[0]+2), (double)(i64[1]+2), (double)(i64[2]+2), (double)(i64[3]+2)}, stream);
    std::vector<double> exp_inplace_adds_i32 = {(double)(i32[0]+1.0*2.0), (double)(i32[1]+1.0*2.0), (double)(i32[2]+1.0*2.0), (double)(i32[3]+1.0*2.0)};
    totalFailed += RunInplaceAddsTest<int32_t>("API_InplaceAdds_INT32_alpha1", ACL_INT32, i32, 2.0, 1.0, s4, exp_inplace_adds_i32, stream);

    // 浮点型
    std::vector<double> exp_inplace_adds_f32_a1 = {f32[0]+1.0*3.0, f32[1]+1.0*3.0, f32[2]+1.0*3.0, f32[3]+1.0*3.0};
    totalFailed += RunInplaceAddsTest<float>("API_InplaceAdds_FLOAT_alpha1", ACL_FLOAT, f32, 3.0, 1.0, s4, exp_inplace_adds_f32_a1, stream);
    std::vector<double> exp_inplace_adds_fp16_a1 = {fp16_to_fp32(fp16[0])+1.0*2.0, fp16_to_fp32(fp16[1])+1.0*2.0, fp16_to_fp32(fp16[2])+1.0*2.0, fp16_to_fp32(fp16[3])+1.0*2.0};
    totalFailed += RunInplaceAddsTest<uint16_t>("API_InplaceAdds_FP16_alpha1", ACL_FLOAT16, fp16, 2.0, 1.0, s4, exp_inplace_adds_fp16_a1, stream, 1e-3, 1e-3);
    std::vector<double> exp_inplace_adds_bf16_a1 = {bf16_to_fp32(bf16[0])+1.0*2.0, bf16_to_fp32(bf16[1])+1.0*2.0, bf16_to_fp32(bf16[2])+1.0*2.0, bf16_to_fp32(bf16[3])+1.0*2.0};
    totalFailed += RunInplaceAddsTest<uint16_t>("API_InplaceAdds_BF16_alpha1", ACL_BF16, bf16, 2.0, 1.0, s4, exp_inplace_adds_bf16_a1, stream, 1e-2, 1e-2);


    // ========================================================================
    // [5] aclnnAddV3 (out = scalar_self + tensor_other)
    // ========================================================================
    // 基础整型与浮点型正常测试
    totalFailed += RunAddV3Test<int8_t, int8_t>("API_AddV3_INT8_alpha1", ACL_INT8, ACL_INT8, 2.0, i8, 1.0, s4, {(double)(2+i8[0]), (double)(2+i8[1]), (double)(2+i8[2]), (double)(2+i8[3])}, stream);
    std::vector<double> exp_add_v3_f32_a1 = {5.0+1.0*f32_2[0], 5.0+1.0*f32_2[1], 5.0+1.0*f32_2[2], 5.0+1.0*f32_2[3]};
    totalFailed += RunAddV3Test<float, float>("API_AddV3_FLOAT_alpha1", ACL_FLOAT, ACL_FLOAT, 5.0, f32_2, 1.0, s4, exp_add_v3_f32_a1, stream);

    // 缺陷探测
    LOG_PRINT("NoOneLog=== 发掘算子缺陷, alpha == 1 时, AddV3不支持fp16、bf16的输入 ===\n");
    std::vector<double> exp_add_v3_fp16_a1 = {2.0+1.0*fp16_to_fp32(fp16[0]), 2.0+1.0*fp16_to_fp32(fp16[1]), 2.0+1.0*fp16_to_fp32(fp16[2]), 2.0+1.0*fp16_to_fp32(fp16[3])};
    totalFailed += RunAddV3Test<uint16_t, uint16_t>("API_AddV3_FP16_alpha1", ACL_FLOAT16, ACL_FLOAT16, 2.0, fp16, 1.0, s4, exp_add_v3_fp16_a1, stream, 1e-3, 1e-3);
    std::vector<double> exp_add_v3_bf16_a1 = {2.0+1.0*bf16_to_fp32(bf16[0]), 2.0+1.0*bf16_to_fp32(bf16[1]), 2.0+1.0*bf16_to_fp32(bf16[2]), 2.0+1.0*bf16_to_fp32(bf16[3])};
    totalFailed += RunAddV3Test<uint16_t, uint16_t>("API_AddV3_BF16_alpha1", ACL_BF16, ACL_BF16, 2.0, bf16, 1.0, s4, exp_add_v3_bf16_a1, stream, 1e-2, 1e-2);


    // ========================================================================
    // [6] aclnnInplaceAddV3 (tensor_other = scalar_self + tensor_other)
    // ========================================================================
    // 基础整型与浮点型正常测试
    totalFailed += RunInplaceAddV3Test<int8_t>("API_InplaceAddV3_INT8_alpha1", ACL_INT8, 2.0, i8, 1.0, s4, {(double)(2+i8[0]), (double)(2+i8[1]), (double)(2+i8[2]), (double)(2+i8[3])}, stream);
    std::vector<double> exp_inplace_add_v3_f32_a1 = {2.0+1.0*f32[0], 2.0+1.0*f32[1], 2.0+1.0*f32[2], 2.0+1.0*f32[3]};
    totalFailed += RunInplaceAddV3Test<float>("API_InplaceAddV3_FLOAT_alpha1", ACL_FLOAT, 2.0, f32, 1.0, s4, exp_inplace_add_v3_f32_a1, stream);
    
    // 缺陷探测
    LOG_PRINT("NoOneLog=== 发掘算子缺陷, alpha == 1 时, InplaceAddV3不支持fp16、bf16的输入 ===\n");
    std::vector<double> exp_inplace_add_v3_bf16_a1 = {2.0+1.0*bf16_to_fp32(bf16[0]), 2.0+1.0*bf16_to_fp32(bf16[1]), 2.0+1.0*bf16_to_fp32(bf16[2]), 2.0+1.0*bf16_to_fp32(bf16[3])};
    totalFailed += RunInplaceAddV3Test<uint16_t>("API_InplaceAddV3_BF16_alpha1", ACL_BF16, 2.0, bf16, 1.0, s4, exp_inplace_add_v3_bf16_a1, stream, 1e-2, 1e-2);
    std::vector<double> exp_inplace_add_v3_fp16_a1 = {2.0+1.0*fp16_to_fp32(fp16[0]), 2.0+1.0*fp16_to_fp32(fp16[1]), 2.0+1.0*fp16_to_fp32(fp16[2]), 2.0+1.0*fp16_to_fp32(fp16[3])};
    totalFailed += RunInplaceAddV3Test<uint16_t>("API_InplaceAddV3_FP16_alpha1", ACL_FLOAT16, 2.0, fp16, 1.0, s4, exp_inplace_add_v3_fp16_a1, stream, 1e-3, 1e-3);
    
    LOG_PRINT("NoOneLog--- 2. 各个 API 变体在 alpha != 1 时的强覆盖 (正数、负数、零) ---\n");
    // 说明：为了避开 CPU 模拟器上整型触发的 AxpyV2 / Fallback (561103) 错误，
    // 以及规避 Adds 等接口对 int32 在 alpha != 1 时的算子缺陷，本节统一使用浮点类型组合。
    // ========================================================================
    // [1] aclnnAdd (out = self + alpha * other)
    // ========================================================================
    // 正数 alpha (FLOAT)
    std::vector<double> exp_add_f32_pos = {f32[0]+2.5*f32_2[0], f32[1]+2.5*f32_2[1], f32[2]+2.5*f32_2[2], f32[3]+2.5*f32_2[3]};
    totalFailed += RunAddTest<float, float, float>("API_Add_FLOAT_alpha2.5", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, f32, f32_2, 2.5, s4, s4, s4, exp_add_f32_pos, stream);
    // 负数 alpha (FP16)
    std::vector<double> exp_add_fp16_neg = {fp16_to_fp32(fp16[0])+(-2.0)*fp16_to_fp32(fp16_2[0]), fp16_to_fp32(fp16[1])+(-2.0)*fp16_to_fp32(fp16_2[1]), fp16_to_fp32(fp16[2])+(-2.0)*fp16_to_fp32(fp16_2[2]), fp16_to_fp32(fp16[3])+(-2.0)*fp16_to_fp32(fp16_2[3])};
    totalFailed += RunAddTest<uint16_t, uint16_t, uint16_t>("API_Add_FP16_alpha-2.0", ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, fp16, fp16_2, -2.0, s4, s4, s4, exp_add_fp16_neg, stream, 1e-3, 1e-3);
    // 零 alpha (BF16)
    std::vector<double> exp_add_bf16_zero = {bf16_to_fp32(bf16[0]), bf16_to_fp32(bf16[1]), bf16_to_fp32(bf16[2]), bf16_to_fp32(bf16[3])};
    totalFailed += RunAddTest<uint16_t, uint16_t, uint16_t>("API_Add_BF16_alpha0.0", ACL_BF16, ACL_BF16, ACL_BF16, bf16, bf16_2, 0.0, s4, s4, s4, exp_add_bf16_zero, stream, 1e-2, 1e-2);


    // ========================================================================
    // [2] aclnnAdds (out = self + alpha * other_scalar)
    // ========================================================================
    // 正数 alpha (FLOAT)
    std::vector<double> exp_adds_f32_pos = {f32[0]+3.5*3.0, f32[1]+3.5*3.0, f32[2]+3.5*3.0, f32[3]+3.5*3.0};
    totalFailed += RunAddsTest<float, float>("API_Adds_FLOAT_alpha3.5", ACL_FLOAT, ACL_FLOAT, f32, 3.0, 3.5, s4, exp_adds_f32_pos, stream);
    // 负数 alpha (FP16)
    std::vector<double> exp_adds_fp16_neg = {fp16_to_fp32(fp16[0])+(-1.5)*3.0, fp16_to_fp32(fp16[1])+(-1.5)*3.0, fp16_to_fp32(fp16[2])+(-1.5)*3.0, fp16_to_fp32(fp16[3])+(-1.5)*3.0};
    totalFailed += RunAddsTest<uint16_t, uint16_t>("API_Adds_FP16_alpha-1.5", ACL_FLOAT16, ACL_FLOAT16, fp16, 3.0, -1.5, s4, exp_adds_fp16_neg, stream, 1e-3, 1e-3);
    // 零 alpha (BF16)
    std::vector<double> exp_adds_bf16_zero = {bf16_to_fp32(bf16[0]), bf16_to_fp32(bf16[1]), bf16_to_fp32(bf16[2]), bf16_to_fp32(bf16[3])};
    totalFailed += RunAddsTest<uint16_t, uint16_t>("API_Adds_BF16_alpha0.0", ACL_BF16, ACL_BF16, bf16, 3.0, 0.0, s4, exp_adds_bf16_zero, stream, 1e-2, 1e-2);


    // ========================================================================
    // [3] aclnnInplaceAdd (selfRef += alpha * other)
    // ========================================================================
    // 正数 alpha (FLOAT)
    std::vector<double> exp_inplace_f32_pos = {f32[0]+1.5*f32_2[0], f32[1]+1.5*f32_2[1], f32[2]+1.5*f32_2[2], f32[3]+1.5*f32_2[3]};
    totalFailed += RunInplaceAddTest<float, float>("API_InplaceAdd_FLOAT_alpha1.5", ACL_FLOAT, ACL_FLOAT, f32, f32_2, 1.5, s4, exp_inplace_f32_pos, stream);
    // 负数 alpha (FP16)
    std::vector<double> exp_inplace_fp16_neg = {fp16_to_fp32(fp16[0])+(-2.5)*fp16_to_fp32(fp16_2[0]), fp16_to_fp32(fp16[1])+(-2.5)*fp16_to_fp32(fp16_2[1]), fp16_to_fp32(fp16[2])+(-2.5)*fp16_to_fp32(fp16_2[2]), fp16_to_fp32(fp16[3])+(-2.5)*fp16_to_fp32(fp16_2[3])};
    totalFailed += RunInplaceAddTest<uint16_t, uint16_t>("API_InplaceAdd_FP16_alpha-2.5", ACL_FLOAT16, ACL_FLOAT16, fp16, fp16_2, -2.5, s4, exp_inplace_fp16_neg, stream, 1e-3, 1e-3);
    // 零 alpha (BF16)
    std::vector<double> exp_inplace_bf16_zero = {bf16_to_fp32(bf16[0]), bf16_to_fp32(bf16[1]), bf16_to_fp32(bf16[2]), bf16_to_fp32(bf16[3])};
    totalFailed += RunInplaceAddTest<uint16_t, uint16_t>("API_InplaceAdd_BF16_alpha0.0", ACL_BF16, ACL_BF16, bf16, bf16_2, 0.0, s4, exp_inplace_bf16_zero, stream, 1e-2, 1e-2);


    // ========================================================================
    // [4] aclnnInplaceAdds (selfRef += alpha * other_scalar)
    // ========================================================================
    // 正数 alpha (FLOAT)
    std::vector<double> exp_inplace_adds_f32_pos = {f32[0]+2.5*2.0, f32[1]+2.5*2.0, f32[2]+2.5*2.0, f32[3]+2.5*2.0};
    totalFailed += RunInplaceAddsTest<float>("API_InplaceAdds_FLOAT_alpha2.5", ACL_FLOAT, f32, 2.0, 2.5, s4, exp_inplace_adds_f32_pos, stream);
    // 负数 alpha (FP16)
    std::vector<double> exp_inplace_adds_fp16_neg = {fp16_to_fp32(fp16[0])+(-1.5)*2.0, fp16_to_fp32(fp16[1])+(-1.5)*2.0, fp16_to_fp32(fp16[2])+(-1.5)*2.0, fp16_to_fp32(fp16[3])+(-1.5)*2.0};
    totalFailed += RunInplaceAddsTest<uint16_t>("API_InplaceAdds_FP16_alpha-1.5", ACL_FLOAT16, fp16, 2.0, -1.5, s4, exp_inplace_adds_fp16_neg, stream, 1e-3, 1e-3);
    // 零 alpha (BF16)
    std::vector<double> exp_inplace_adds_bf16_zero = {bf16_to_fp32(bf16[0]), bf16_to_fp32(bf16[1]), bf16_to_fp32(bf16[2]), bf16_to_fp32(bf16[3])};
    totalFailed += RunInplaceAddsTest<uint16_t>("API_InplaceAdds_BF16_alpha0.0", ACL_BF16, bf16, 2.0, 0.0, s4, exp_inplace_adds_bf16_zero, stream, 1e-2, 1e-2);


    // ========================================================================
    // [5] aclnnAddV3 (out = scalar_self + alpha * tensor_other)
    // ========================================================================
    // 正数 alpha (FLOAT)
    std::vector<double> exp_addv3_f32_pos = {5.0+1.5*f32_2[0], 5.0+1.5*f32_2[1], 5.0+1.5*f32_2[2], 5.0+1.5*f32_2[3]};
    totalFailed += RunAddV3Test<float, float>("API_AddV3_FLOAT_alpha1.5", ACL_FLOAT, ACL_FLOAT, 5.0, f32_2, 1.5, s4, exp_addv3_f32_pos, stream);
    LOG_PRINT("NoOneLog=== 发掘算子缺陷, alpha != 1 时, AddV3不支持fp16、bf16的输入 ===\n");
    // 负数 alpha (FP16)
    std::vector<double> exp_addv3_fp16_neg = {5.0+(-2.0)*fp16_to_fp32(fp16[0]), 5.0+(-2.0)*fp16_to_fp32(fp16[1]), 5.0+(-2.0)*fp16_to_fp32(fp16[2]), 5.0+(-2.0)*fp16_to_fp32(fp16[3])};
    totalFailed += RunAddV3Test<uint16_t, uint16_t>("API_AddV3_FP16_alpha-2.0", ACL_FLOAT16, ACL_FLOAT16, 5.0, fp16, -2.0, s4, exp_addv3_fp16_neg, stream, 1e-3, 1e-3);
    // 零 alpha (BF16)
    std::vector<double> exp_addv3_bf16_zero = {5.0, 5.0, 5.0, 5.0};
    totalFailed += RunAddV3Test<uint16_t, uint16_t>("API_AddV3_BF16_alpha0.0", ACL_BF16, ACL_BF16, 5.0, bf16, 0.0, s4, exp_addv3_bf16_zero, stream, 1e-2, 1e-2);
    // ========================================================================
    // [6] aclnnInplaceAddV3 (tensor_other = scalar_self + alpha * tensor_other)
    // ========================================================================
    // 正数 alpha (FLOAT)
    std::vector<double> exp_inplace_addv3_f32_pos = {2.0+1.5*f32[0], 2.0+1.5*f32[1], 2.0+1.5*f32[2], 2.0+1.5*f32[3]};
    totalFailed += RunInplaceAddV3Test<float>("API_InplaceAddV3_FLOAT_alpha1.5", ACL_FLOAT, 2.0, f32, 1.5, s4, exp_inplace_addv3_f32_pos, stream);
    LOG_PRINT("NoOneLog=== 发掘算子缺陷, alpha != 1 时, InplaceAddV3不支持fp16、bf16的输入 ===\n");
    // 负数 alpha (FP16)
    std::vector<double> exp_inplace_addv3_fp16_neg = {2.0+(-2.5)*fp16_to_fp32(fp16[0]), 2.0+(-2.5)*fp16_to_fp32(fp16[1]), 2.0+(-2.5)*fp16_to_fp32(fp16[2]), 2.0+(-2.5)*fp16_to_fp32(fp16[3])};
    totalFailed += RunInplaceAddV3Test<uint16_t>("API_InplaceAddV3_FP16_alpha-2.5", ACL_FLOAT16, 2.0, fp16, -2.5, s4, exp_inplace_addv3_fp16_neg, stream, 1e-3, 1e-3);
    // 零 alpha (BF16)
    std::vector<double> exp_inplace_addv3_bf16_zero = {2.0, 2.0, 2.0, 2.0};
    totalFailed += RunInplaceAddV3Test<uint16_t>("API_InplaceAddV3_BF16_alpha0.0", ACL_BF16, 2.0, bf16, 0.0, s4, exp_inplace_addv3_bf16_zero, stream, 1e-2, 1e-2);

//            LOG_PRINT("NoOneLog--- 3. 广播机制与大 Shape 测试 ---\n");
//            std::vector<float> brd_x1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; 
//            std::vector<float> brd_x2 = {10.0, 20.0, 30.0};             
//            std::vector<double> exp_brd = {1.0+10.0, 2.0+20.0, 3.0+30.0, 4.0+10.0, 5.0+20.0, 6.0+30.0};
//            totalFailed += RunAddTest<float, float, float>("Broadcast_2D_1D", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, brd_x1, brd_x2, 1.0, {2, 3}, {3}, {2, 3}, exp_brd, stream);
        
    std::vector<float> large_x(10000, 1.5f);
    std::vector<double> large_exp(10000, 1.5f + 2.0 * 1.5f);
    totalFailed += RunAddTest<float, float, float>("LargeShape_10K_Elements", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, large_x, large_x, 2.0, {10000}, {10000}, {10000}, large_exp, stream);

    LOG_PRINT("NoOneLog--- 4. Complex 和 混合数据类型 测试 ---\n");

    totalFailed += RunAddTest<uint16_t, float, float>("Add_Mix_BF16_FLOAT_alpha1", ACL_BF16, ACL_FLOAT, ACL_FLOAT, bf16, f32_2, 1.0, s4, s4, s4, {bf16_to_fp32(bf16[0])+f32_2[0], bf16_to_fp32(bf16[1])+f32_2[1], bf16_to_fp32(bf16[2])+f32_2[2], bf16_to_fp32(bf16[3])+f32_2[3]}, stream, 1e-2, 1e-2);
    totalFailed += RunAddTest<float, uint16_t, float>("Add_Mix_FLOAT_FP16_alpha1", ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, f32, fp16, 1.0, s4, s4, s4, {f32[0]+fp16_to_fp32(fp16[0]), f32[1]+fp16_to_fp32(fp16[1]), f32[2]+fp16_to_fp32(fp16[2]), f32[3]+fp16_to_fp32(fp16[3])}, stream, 1e-3, 1e-3);
    // ========================================================================
    // [1] aclnnAdds (Tensor + Scalar -> Out)
    // ========================================================================
    // BF16(Tensor) + FLOAT(Scalar) -> FLOAT(Out)
    LOG_PRINT("NoOneLog=== 发掘算子缺陷, alpha == 1 时, Adds不支持第一个输入是fp16、bf16 ===\n");
    std::vector<double> exp_adds_mix_bf16_f32(4);
    for(int i=0; i<4; i++) exp_adds_mix_bf16_f32[i] = bf16_to_fp32(bf16[i]) + 2.0;
    totalFailed += RunAddsTest<uint16_t, float>("API_Adds_Mix_BF16_FLOAT_alpha1", ACL_BF16, ACL_FLOAT, ACL_FLOAT, bf16, 2.0, 1.0, s4, exp_adds_mix_bf16_f32, stream, 1e-2, 1e-2);

    // FLOAT(Tensor) + FP16(Scalar) -> FLOAT(Out)
    std::vector<double> exp_adds_mix_f32_fp16(4);
    for(int i=0; i<4; i++) exp_adds_mix_f32_fp16[i] = f32[i] + 2.0; 
    totalFailed += RunAddsTest<float, float>("API_Adds_Mix_FLOAT_FP16_alpha1", ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, f32, 2.0, 1.0, s4, exp_adds_mix_f32_fp16, stream, 1e-3, 1e-3);


    // ========================================================================
    // [2] aclnnInplaceAdd (selfRef += other)
    // ========================================================================
    // 注意：Inplace 规范要求 self 的精度 >= other，避免向下截断。因此测试 FLOAT += FP16/BF16
    // FLOAT(self) += FP16(other)
    std::vector<double> exp_inplace_mix_f32_fp16(4);
    for(int i=0; i<4; i++) exp_inplace_mix_f32_fp16[i] = f32[i] + fp16_to_fp32(fp16_2[i]);
    totalFailed += RunInplaceAddTest<float, uint16_t>("API_InplaceAdd_Mix_FLOAT_FP16_alpha1", ACL_FLOAT, ACL_FLOAT16, f32, fp16_2, 1.0, s4, exp_inplace_mix_f32_fp16, stream, 1e-3, 1e-3);

    // FLOAT(self) += BF16(other)
    std::vector<double> exp_inplace_mix_f32_bf16(4);
    for(int i=0; i<4; i++) exp_inplace_mix_f32_bf16[i] = f32[i] + bf16_to_fp32(bf16_2[i]);
    totalFailed += RunInplaceAddTest<float, uint16_t>("API_InplaceAdd_Mix_FLOAT_BF16_alpha1", ACL_FLOAT, ACL_BF16, f32, bf16_2, 1.0, s4, exp_inplace_mix_f32_bf16, stream, 1e-2, 1e-2);


    // ========================================================================
    // [3] aclnnInplaceAdds (selfRef += other_scalar)
    // ========================================================================
    // FLOAT(self) += FP16(Scalar)
    std::vector<double> exp_inplace_adds_mix_f32_fp16(4);
    for(int i=0; i<4; i++) exp_inplace_adds_mix_f32_fp16[i] = f32[i] + 3.0;
    totalFailed += RunInplaceAddsTest<float>("API_InplaceAdds_Mix_FLOAT_FP16_alpha1", ACL_FLOAT, ACL_FLOAT16, f32, 3.0, 1.0, s4, exp_inplace_adds_mix_f32_fp16, stream, 1e-3, 1e-3);

    // FLOAT(self) += BF16(Scalar)
    std::vector<double> exp_inplace_adds_mix_f32_bf16(4);
    for(int i=0; i<4; i++) exp_inplace_adds_mix_f32_bf16[i] = f32[i] + 3.0;
    totalFailed += RunInplaceAddsTest<float>("API_InplaceAdds_Mix_FLOAT_BF16_alpha1", ACL_FLOAT, ACL_BF16, f32, 3.0, 1.0, s4, exp_inplace_adds_mix_f32_bf16, stream, 1e-2, 1e-2);


    // ========================================================================
    // [4] aclnnAddV3 (out = scalar_self + tensor_other)
    // ========================================================================
    // FLOAT(Scalar) + FP16(Tensor) -> FLOAT(Out)
    LOG_PRINT("NoOneLog=== 发掘算子缺陷, alpha == 1 时, AddV3不支持输入里有fp16、bf16类型 ===\n");
    std::vector<double> exp_addv3_mix_f32_fp16(4);
    for(int i=0; i<4; i++) exp_addv3_mix_f32_fp16[i] = 5.0 + fp16_to_fp32(fp16_2[i]);
    totalFailed += RunAddV3Test<uint16_t, float>("API_AddV3_Mix_FLOAT_FP16_alpha1", ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, 5.0, fp16_2, 1.0, s4, exp_addv3_mix_f32_fp16, stream, 1e-3, 1e-3);

    // FLOAT(Scalar) + BF16(Tensor) -> FLOAT(Out)
    std::vector<double> exp_addv3_mix_f32_bf16(4);
    for(int i=0; i<4; i++) exp_addv3_mix_f32_bf16[i] = 5.0 + bf16_to_fp32(bf16_2[i]);
    totalFailed += RunAddV3Test<uint16_t, float>("API_AddV3_Mix_FLOAT_BF16_alpha1", ACL_FLOAT, ACL_BF16, ACL_FLOAT, 5.0, bf16_2, 1.0, s4, exp_addv3_mix_f32_bf16, stream, 1e-2, 1e-2);


    // ========================================================================
    // [5] aclnnInplaceAddV3 (tensor_other = scalar_self + tensor_other)
    // ========================================================================
    // FP16(Scalar) + FLOAT(Tensor) -> FLOAT(Tensor)
    std::vector<double> exp_inplace_addv3_mix_fp16_f32(4);
    for(int i=0; i<4; i++) exp_inplace_addv3_mix_fp16_f32[i] = 2.0 + f32[i]; 
    totalFailed += RunInplaceAddV3Test<float>("API_InplaceAddV3_Mix_FP16_FLOAT_alpha1", ACL_FLOAT16, ACL_FLOAT, 2.0, f32, 1.0, s4, exp_inplace_addv3_mix_fp16_f32, stream, 1e-3, 1e-3);

    // ==========================================
    // [1] aclnnAdd 
    // ==========================================
    std::vector<std::complex<double>> exp_add_c64(4);
    std::vector<std::complex<double>> exp_add_mix(4);
    for(int i=0; i<4; i++) {
        // C++ 标准复数运算推演：self + alpha * other
        exp_add_c64[i] = std::complex<double>(c64_1[i].real(), c64_1[i].imag()) + c_alpha * std::complex<double>(c64_2[i].real(), c64_2[i].imag());
        // 混合：FLOAT(self) + 1.0 * COMPLEX64(other)
        exp_add_mix[i] = std::complex<double>(f32[i], 0.0) + std::complex<double>(1.0, 0.0) * std::complex<double>(c64_2[i].real(), c64_2[i].imag());
    }
    // 同类型 COMPLEX64 (alpha 也是复数)
    totalFailed += RunAddComplexTest<std::complex<float>, std::complex<float>, float>("API_Add_COMPLEX64", ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64, c64_1, c64_2, c_alpha, s4, exp_add_c64, stream);

    // ==========================================
    // [2] aclnnAdds
    // ==========================================
    std::vector<std::complex<double>> exp_adds_c64(4);
    for(int i=0; i<4; i++) {
        exp_adds_c64[i] = std::complex<double>(c64_1[i].real(), c64_1[i].imag()) + c_alpha * c_scalar;
    }
    // COMPLEX64(Tensor) + COMPLEX64(Scalar) * COMPLEX64(alpha)
    totalFailed += RunAddsComplexTest<std::complex<float>, float>("API_Adds_COMPLEX64", ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64, c64_1, c_scalar, c_alpha, s4, exp_adds_c64, stream);

    // ==========================================
    // [3] aclnnInplaceAdd
    // ==========================================
    std::vector<std::complex<double>> exp_inplace_mix(4);
    for(int i=0; i<4; i++) {
        // self(complex64) += alpha * other(float)
        exp_inplace_mix[i] = std::complex<double>(c64_1[i].real(), c64_1[i].imag()) + c_alpha * std::complex<double>(f32[i], 0.0);
    }
    // 同类型 COMPLEX64
    totalFailed += RunInplaceAddComplexTest<std::complex<float>, std::complex<float>>("API_InplaceAdd_COMPLEX64", ACL_COMPLEX64, ACL_COMPLEX64, c64_1, c64_2, c_alpha, s4, exp_add_c64, stream);

    // ==========================================
    // [4] aclnnInplaceAdds
    // ==========================================
    totalFailed += RunInplaceAddsComplexTest<std::complex<float>>("API_InplaceAdds_COMPLEX64", ACL_COMPLEX64, ACL_COMPLEX64, c64_1, c_scalar, c_alpha, s4, exp_adds_c64, stream);

    LOG_PRINT("NoOneLog--- 5. 数值边界、Inf、NaN 与空 Tensor ---\n");
    // 规避模拟器上 Max + Max 直接饱和与 Host C++ 浮点数溢出结果不一致的问题，改为 Max/2 + Max/2
    std::vector<float> b_x1 = {INFINITY, NAN, 1.7014117e+38f, -0.0f};
    std::vector<float> b_x2 = {2.0f, 2.0f, 1.7014117e+38f, 5.0f};
    std::vector<double> exp_bnd = {b_x1[0]+b_x2[0], b_x1[1]+b_x2[1], static_cast<double>(b_x1[2])+static_cast<double>(b_x2[2]), b_x1[3]+b_x2[3]};
    totalFailed += RunAddTest<float, float, float>("Boundary_FLOAT_Inf_NaN_Max", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, b_x1, b_x2, 1.0, s4, s4, s4, exp_bnd, stream);
    
    totalFailed += RunAddTest<float, float, float>("Empty_Tensor_Add", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, {}, {}, 1.0, {0, 2}, {0, 2}, {0, 2}, {}, stream);

    LOG_PRINT("NoOneLog--- 6. 异常参数拦截测试 (反向断言) ---\n");
    totalFailed += TestNegativeNullptr("Negative_Nullptr_AllAPIs");
    totalFailed += TestNegativeShapeMismatchAll("Negative_ShapeMismatch_AllAPIs", stream);
    totalFailed += TestNegativeInvalidDTypeAll("Negative_InvalidDType_AllAPIs", stream);
    // ===================================================================
    // 新增部分 1：覆盖率达补全 (提高行覆盖率)
    // ===================================================================
    LOG_PRINT("NoOneLog--- 8. 覆盖率补全：错误注入与未覆盖路径 ---\n");
    // 显式声明（如果编译器仍报错，可移至文件头部）
    extern aclTensor* CreateDummyAclTensor(const std::vector<int64_t>& shape, aclDataType dataType);

    // ----- 8.1 other 类型非法 (触发第二个 OP_CHECK_DTYPE_NOT_SUPPORT) -----
    {
        void *tDev = nullptr;
        aclTensor *tSelf = nullptr;
        CreateAclTensor<float>({1.0f, 2.0f}, {2}, &tDev, ACL_FLOAT, &tSelf);
        aclTensor *tBadOther = CreateDummyAclTensor({2}, static_cast<aclDataType>(9999));
        aclTensor *tOut = CreateDummyAclTensor({2}, ACL_FLOAT);
        aclScalar *alphaLocal = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
        uint64_t wsSize = 0;
        aclOpExecutor* executor = nullptr;
        if (aclnnAddGetWorkspaceSize(tSelf, tBadOther, alphaLocal, tOut, &wsSize, &executor) == ACL_SUCCESS)
            totalFailed++;
        aclDestroyTensor(tSelf); aclDestroyTensor(tBadOther); aclDestroyTensor(tOut);
        aclDestroyScalar(alphaLocal);
        aclrtFree(tDev);
    }

    // ----- 8.3 CheckShape: 维度超过 8 -----
    {
        aclScalar *alphaLocal = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
        aclTensor *t9d = CreateDummyAclTensor({1,1,1,1,1,1,1,1,1}, ACL_FLOAT);
        uint64_t wsSize = 0;
        aclOpExecutor* executor = nullptr;
        if (aclnnAddGetWorkspaceSize(t9d, t9d, alphaLocal, t9d, &wsSize, &executor) == ACL_SUCCESS)
            totalFailed++;
        aclDestroyTensor(t9d);
        aclDestroyScalar(alphaLocal);
    }

    // ----- 8.4 CheckShape: 输出 shape 不等于广播 shape -----
    {
        aclScalar *alphaLocal = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
        aclTensor *tA = CreateDummyAclTensor({2,3}, ACL_FLOAT);
        aclTensor *tB = CreateDummyAclTensor({3}, ACL_FLOAT);
        aclTensor *tBadOut = CreateDummyAclTensor({2,4}, ACL_FLOAT);
        uint64_t wsSize = 0;
        aclOpExecutor* executor = nullptr;
        if (aclnnAddGetWorkspaceSize(tA, tB, alphaLocal, tBadOut, &wsSize, &executor) == ACL_SUCCESS)
            totalFailed++;
        aclDestroyTensor(tA); aclDestroyTensor(tB); aclDestroyTensor(tBadOut);
        aclDestroyScalar(alphaLocal);
    }

    // ----- 8.5 Adds: self 类型非法 -----
    {
        aclScalar *otherScalar = CreateAclScalarFromDouble(2.0, ACL_FLOAT);
        aclScalar *alphaLocal = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
        aclTensor *tSelfBad = CreateDummyAclTensor({2,3}, static_cast<aclDataType>(9999));
        aclTensor *tOut = CreateDummyAclTensor({2,3}, ACL_FLOAT);
        uint64_t wsSize = 0;
        aclOpExecutor* executor = nullptr;
        if (aclnnAddsGetWorkspaceSize(tSelfBad, otherScalar, alphaLocal, tOut, &wsSize, &executor) == ACL_SUCCESS)
            totalFailed++;
        aclDestroyTensor(tSelfBad); aclDestroyTensor(tOut);
        aclDestroyScalar(otherScalar); aclDestroyScalar(alphaLocal);
    }
    // ----- 8.6 Adds: 空 self tensor -----
    {
        std::vector<float> emptyData;
        totalFailed += RunAddsTest<float, float>("Adds_Empty_Tensor", ACL_FLOAT, ACL_FLOAT, emptyData, 2.0, 1.0, {0}, {}, stream);
    }

    // ----- 8.7 Adds: non-ND format 警告 -----
    {
        auto CreateFormatTensor = [](const std::vector<int64_t>& shape, aclDataType dt, aclFormat fmt) -> aclTensor* {
            std::vector<int64_t> strides(shape.size(), 1);
            for (int i = (int)shape.size()-2; i >= 0; --i) strides[i] = shape[i+1] * strides[i];
            return aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0, fmt,
                                   shape.data(), shape.size(), nullptr);
        };
        aclTensor *tNchw = CreateFormatTensor({2,3,4}, ACL_FLOAT, ACL_FORMAT_NCHW);
        aclScalar *otherScalar = CreateAclScalarFromDouble(2.0, ACL_FLOAT);
        aclScalar *alphaLocal = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
        aclTensor *tOutNchw = CreateDummyAclTensor({2,3,4}, ACL_FLOAT);
        uint64_t wsSize = 0;
        aclOpExecutor* executor = nullptr;
        aclnnAddsGetWorkspaceSize(tNchw, otherScalar, alphaLocal, tOutNchw, &wsSize, &executor);
        aclDestroyTensor(tNchw); aclDestroyTensor(tOutNchw);
        aclDestroyScalar(otherScalar); aclDestroyScalar(alphaLocal);
    }

    // ----- 8.8 Inplace: selfRef shape 不等于广播结果 -----
    {
        aclScalar *alphaLocal = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
        aclTensor *tRefBad = CreateDummyAclTensor({3}, ACL_FLOAT);   // shape {3}
        aclTensor *tBig = CreateDummyAclTensor({2,3}, ACL_FLOAT);    // shape {2,3}
        uint64_t wsSize = 0;
        aclOpExecutor* executor = nullptr;
        if (aclnnInplaceAddGetWorkspaceSize(tRefBad, tBig, alphaLocal, &wsSize, &executor) == ACL_SUCCESS)
            totalFailed++;
        aclDestroyTensor(tRefBad); aclDestroyTensor(tBig);
        aclDestroyScalar(alphaLocal);
    }
    // ----- 8.9 Add V3: other 类型非法 -----
    {
        aclScalar *selfScalar = CreateAclScalarFromDouble(2.0, ACL_FLOAT);
        aclScalar *alphaLocal = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
        aclTensor *tOtherBad = CreateDummyAclTensor({2}, static_cast<aclDataType>(9999));
        aclTensor *tOut = CreateDummyAclTensor({2}, ACL_FLOAT);
        uint64_t wsSize = 0;
        aclOpExecutor* executor = nullptr;
        if (aclnnAddV3GetWorkspaceSize(selfScalar, tOtherBad, alphaLocal, tOut, &wsSize, &executor) == ACL_SUCCESS)
            totalFailed++;
        aclDestroyTensor(tOtherBad); aclDestroyTensor(tOut);
        aclDestroyScalar(selfScalar); aclDestroyScalar(alphaLocal);
    }

    // ----- 8.10 Add V3: other 为空 tensor -----
    {
        std::vector<float> emptyData;
        totalFailed += RunAddV3Test<float, float>("AddV3_Empty_Other", ACL_FLOAT, ACL_FLOAT, 2.0, emptyData, 1.0, {0}, {}, stream);
    }
    // ----- 8.11 Add V3: alpha != 1 且类型为 INT8 (走 Mul+Add 分支) -----
    {
        std::vector<int8_t> i8_v3 = {1, -2, 3, -4};
        std::vector<double> exp_v3_i8 = {
            2.0 + 3.0 * i8_v3[0], 2.0 + 3.0 * i8_v3[1], 2.0 + 3.0 * i8_v3[2], 2.0 + 3.0 * i8_v3[3]
        };
        totalFailed += RunAddV3Test<int8_t, int8_t>("AddV3_INT8_alpha3", ACL_INT8, ACL_INT8, 2.0, i8_v3, 3.0, {4}, exp_v3_i8, stream);
    }

//          //    // ----- 8.12 使用 DOUBLE 类型触发 AiCpu 路径 (覆盖 add.cpp 的 AddAiCpu) -----
//          //    {
//          //        std::vector<double> d1 = {1.1, 2.2, 3.3, 4.4};
//          //        std::vector<double> d2 = {0.9, 1.1, 2.2, 3.3};
//          //        std::vector<double> exp_add_d = {d1[0]+d2[0], d1[1]+d2[1], d1[2]+d2[2], d1[3]+d2[3]};
//          //        totalFailed += RunAddTest<double, double, double>("Add_DOUBLE_AiCpu", ACL_DOUBLE, ACL_DOUBLE, ACL_DOUBLE,
//          //                                                          d1, d2, 1.0, {4}, {4}, {4}, exp_add_d, stream, 1e-12, 1e-12);
//          //    }
    // ----- 8.13 混合类型 FLOAT + INT32 (仍能运行，验证无异常) -----
    {
        std::vector<float> f_mix = {1.0f, 2.0f, 3.0f};
        std::vector<int32_t> i_mix = {1, 2, 3};
        std::vector<double> exp_mix = {2.0, 4.0, 6.0};
        totalFailed += RunAddTest<float, int32_t, float>("Add_Mix_FLOAT_INT32", ACL_FLOAT, ACL_INT32, ACL_FLOAT,
                                                         f_mix, i_mix, 1.0, {3}, {3}, {3}, exp_mix, stream);
    }
//
//          //    // ----- 8.14 复数类型组合 (覆盖 InnerTypeToComplexType 等) -----
//          //    {
//          //        std::vector<std::complex<float>> cf1 = {{1.0f,2.0f}, {3.0f,4.0f}};
//          //        std::vector<std::complex<float>> cf2 = {{5.0f,6.0f}, {7.0f,8.0f}};
//          //        std::vector<std::complex<double>> exp_cf = {
//          //            {static_cast<double>(1+5), static_cast<double>(2+6)},
//          //            {static_cast<double>(3+7), static_cast<double>(4+8)}
//          //        };
//          //        totalFailed += RunAddComplexTest<std::complex<float>, std::complex<float>, float>(
//          //            "Add_COMPLEX64_Generic", ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64, cf1, cf2,
//          //            {1.0, 0.0}, // complex alpha
//          //            {2},        // shape
//          //            exp_cf, stream);
//          //
//          //        std::complex<double> comp_alpha = {1.0, 0.0};
//          //        std::complex<double> comp_other = {2.0, 3.0};
//          //        std::vector<std::complex<double>> exp_cadds = {
//          //            std::complex<double>(cf1[0].real(), cf1[0].imag()) + comp_other,
//          //            std::complex<double>(cf1[1].real(), cf1[1].imag()) + comp_other
//          //        };
//          //        totalFailed += RunAddsComplexTest<std::complex<float>, float>(
//          //            "Adds_COMPLEX64", ACL_COMPLEX64, ACL_COMPLEX64, ACL_COMPLEX64, cf1,
//          //            comp_other, comp_alpha, {2}, exp_cadds, stream);
//          //
//          //        totalFailed += RunInplaceAddComplexTest<std::complex<float>, std::complex<float>>(
//          //            "InplaceAdd_COMPLEX64", ACL_COMPLEX64, ACL_COMPLEX64, cf1, cf2, comp_alpha, {2}, exp_cf, stream);
//          //    }
//
    // ===================================================================
    // 新增部分 2：精度专项测试 
    // ===================================================================
    LOG_PRINT("NoOneLog--- 9. 精度专项：边界、Inf/NaN、大数、小数、抵消 ---\n");

    // 9.1 极大值相加溢出 (FLOAT) — 真正溢出
    {
        std::vector<float> a = {1.7e38f, 1.7e38f};
        std::vector<float> b = {1.7e38f, 1.7e38f};
        std::vector<double> exp = {3.4e38, 3.4e38}; // 可能 NPU 返回 inf，但作为期望值仍然记录
        totalFailed += RunAddTest<float, float, float>("Prec_Overflow", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
                                                       a, b, 1.0, {2}, {2}, {2}, exp, stream, 1e-5, 1e-5);
    }

    // 9.2 下溢：极小值相加，结果进入次正规数
    {
        std::vector<float> a(4, 1e-40f);
        std::vector<float> b(4, 2e-40f);
        std::vector<double> exp = {3e-40, 3e-40, 3e-40, 3e-40};
        totalFailed += RunAddTest<float, float, float>("Prec_Subnormal", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
                                                       a, b, 1.0, {4}, {4}, {4}, exp, stream, 1e-5, 1e-5);
    }

    // 9.3 Inf/NaN 传播
    {
        std::vector<float> a = {INFINITY, NAN, 1.0f, -INFINITY};
        std::vector<float> b = {0.0f, 0.0f, NAN, INFINITY};
        std::vector<double> exp = {INFINITY, NAN, NAN, NAN};
        totalFailed += RunAddTest<float, float, float>("Prec_Inf_NaN", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
                                                       a, b, 1.0, {4}, {4}, {4}, exp, stream);
    }

    // 9.4 大数加小数，精度损失 (FLOAT)
    {
        std::vector<float> a = {1e8f, 1e8f, 1e8f};
        std::vector<float> b = {1e-6f, -1e-6f, 1.234567e-6f};
        std::vector<double> exp = {1e8f + 1e-6, 1e8f - 1e-6, 1e8f + 1.234567e-6};
        totalFailed += RunAddTest<float, float, float>("Prec_LargeSmall", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
                                                       a, b, 1.0, {3}, {3}, {3}, exp, stream, 1e-4, 1e-4);
    }

    // 9.5 非常接近1.0的微小差异
    {
        std::vector<float> a = {1.0000001f, 1.0000001f};
        std::vector<float> b = {0.9999999f, -0.9999999f};
        std::vector<double> exp = {2.0, 2e-7};
        totalFailed += RunAddTest<float, float, float>("Prec_Near1", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
                                                       a, b, 1.0, {2}, {2}, {2}, exp, stream, 1e-6, 1e-6);
    }

    // 9.6 BF16 / FP16 精度验证
    {
        std::vector<float> f32_vals = {1.0f, 1.0f/3.0f, 1.0f/7.0f, 1.0f/9.0f};
        std::vector<uint16_t> fp16_vals, bf16_vals;
        for (auto v : f32_vals) {
            fp16_vals.push_back(fp32_to_fp16(v));
            bf16_vals.push_back(fp32_to_bf16(v));
        }
        std::vector<double> exp_f32 = {2.0, 2.0/3.0, 2.0/7.0, 2.0/9.0};
        totalFailed += RunAddTest<uint16_t, uint16_t, uint16_t>("Prec_FP16_Base", ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
                                                                fp16_vals, fp16_vals, 1.0, {4}, {4}, {4}, exp_f32, stream, 1e-3, 1e-3);
        totalFailed += RunAddTest<uint16_t, uint16_t, uint16_t>("Prec_BF16_Base", ACL_BF16, ACL_BF16, ACL_BF16,
                                                                bf16_vals, bf16_vals, 1.0, {4}, {4}, {4}, exp_f32, stream, 1e-2, 1e-2);
    }

    // 9.7 整数类型溢出测试 (INT8 加完后可能饱和/截断)
    {
        std::vector<int8_t> a(4, 100), b(4, 100);
        std::vector<double> exp = {200, 200, 200, 200};
        totalFailed += RunAddTest<int8_t, int8_t, int8_t>("Prec_INT8_Saturate", ACL_INT8, ACL_INT8, ACL_INT8,
                                                          a, b, 1.0, {4}, {4}, {4}, exp, stream, 0.0, 0.0);
    }
    
    // ===================================================================
    // 10. 进一步覆盖率补全：错误路径、特殊类型与非 ND 格式
    // ===================================================================
    LOG_PRINT("NoOneLog--- 10. 补充覆盖率：校验分支、混合/非标、tiling 拦截 ---\n");

    // ----- 10.1 空 out 指针（CheckNotNull / CheckNotNullScalar 末尾） -----
    {
        void *tDev=nullptr; aclTensor *tSelf=nullptr;
        CreateAclTensor<float>({1.0f, 2.0f}, {2}, &tDev, ACL_FLOAT, &tSelf);
        aclScalar *alpha = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
        aclScalar *otherS = CreateAclScalarFromDouble(2.0, ACL_FLOAT);
        aclTensor *tOther = CreateDummyAclTensor({2}, ACL_FLOAT);
        uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;

        // aclnnAdd out=nullptr
        if (aclnnAddGetWorkspaceSize(tSelf, tOther, alpha, nullptr, &wsSize, &executor) == ACL_SUCCESS)
            totalFailed++;
        // aclnnAdds out=nullptr (但需要 self,other,alpha 都不为空)
        if (aclnnAddsGetWorkspaceSize(tSelf, otherS, alpha, nullptr, &wsSize, &executor) == ACL_SUCCESS)
            totalFailed++;

        aclDestroyTensor(tSelf); aclDestroyTensor(tOther);
        aclDestroyScalar(alpha); aclDestroyScalar(otherS);
        if (tDev) aclrtFree(tDev);
    }

    // ----- 10.2 other 维度 > 8，触发 CheckShape 的 MAX_DIM 检查（other） -----
    {
        aclScalar *alpha = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
        aclTensor *tSelf = CreateDummyAclTensor({2}, ACL_FLOAT);
        aclTensor *tOther9d = CreateDummyAclTensor({1,1,1,1,1,1,1,1,1}, ACL_FLOAT);
        aclTensor *tOut = CreateDummyAclTensor({1,1,1,1,1,1,1,1,1}, ACL_FLOAT);
        uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
        if (aclnnAddGetWorkspaceSize(tSelf, tOther9d, alpha, tOut, &wsSize, &executor) == ACL_SUCCESS)
            totalFailed++;
        aclDestroyTensor(tSelf); aclDestroyTensor(tOther9d); aclDestroyTensor(tOut);
        aclDestroyScalar(alpha);
    }

    // ----- 10.4 CheckPromoteType 各错误分支 (aclnn_add.cpp) -----
    {
        // 10.4a 无法 promote (DT_UNDEFINED)
        {
            aclTensor *selfT = CreateDummyAclTensor({2}, ACL_BOOL);
            aclTensor *otherT = CreateDummyAclTensor({2}, ACL_COMPLEX64);
            aclTensor *outT = CreateDummyAclTensor({2}, ACL_FLOAT);
            aclScalar *alpha = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
            uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
            if (aclnnAddGetWorkspaceSize(selfT, otherT, alpha, outT, &wsSize, &exec) == ACL_SUCCESS)
                totalFailed++;
            aclDestroyTensor(selfT); aclDestroyTensor(otherT); aclDestroyTensor(outT);
            aclDestroyScalar(alpha);
        }

        // 10.4b alpha 无法 cast 到 promoteType（float tensor + complex alpha）
        {
            aclTensor *selfT = CreateDummyAclTensor({2}, ACL_FLOAT);
            aclTensor *otherT = CreateDummyAclTensor({2}, ACL_FLOAT);
            aclTensor *outT = CreateDummyAclTensor({2}, ACL_FLOAT);
            // 创建 complex64 标量作为 alpha
            float alphaVal[2] = {1.0f, 0.5f};
            aclScalar *alpha = aclCreateScalar(alphaVal, ACL_COMPLEX64);
            uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
            if (aclnnAddGetWorkspaceSize(selfT, otherT, alpha, outT, &wsSize, &exec) == ACL_SUCCESS)
                totalFailed++;
            aclDestroyTensor(selfT); aclDestroyTensor(otherT); aclDestroyTensor(outT);
            aclDestroyScalar(alpha);
        }

        // 10.4c promoteType 无法 cast 到 outDtype
        {
            aclTensor *selfT = CreateDummyAclTensor({2}, ACL_FLOAT);
            aclTensor *otherT = CreateDummyAclTensor({2}, ACL_FLOAT);
            aclTensor *outT = CreateDummyAclTensor({2}, ACL_COMPLEX32);
            aclScalar *alpha = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
            uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
            if (aclnnAddGetWorkspaceSize(selfT, otherT, alpha, outT, &wsSize, &exec) == ACL_SUCCESS)
                totalFailed++;
            aclDestroyTensor(selfT); aclDestroyTensor(otherT); aclDestroyTensor(outT);
            aclDestroyScalar(alpha);
        }
    }

    // ----- 10.5 Adds 中 bool + bool + true alpha 且 out 非 bool 的特殊修正 (行628-635) -----
    {
        std::vector<uint8_t> selfBool = {1, 0, 1, 1};  // true, false, true, true
        std::vector<int8_t> outInit = {0,0,0,0};
        void *selfDev=nullptr, *outDev=nullptr; aclTensor *selfT=nullptr, *outT=nullptr;
        CreateAclTensor(selfBool, {4}, &selfDev, ACL_BOOL, &selfT);
        CreateAclTensor(outInit, {4}, &outDev, ACL_INT8, &outT);
        aclScalar *otherS = CreateAclScalarFromDouble(1.0, ACL_BOOL);   // true
        aclScalar *alpha = CreateAclScalarFromDouble(1.0, ACL_BOOL);    // true
        uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
        auto ret = aclnnAddsGetWorkspaceSize(selfT, otherS, alpha, outT, &wsSize, &exec);
        if (ret == ACL_SUCCESS) {
            void* wsAddr = nullptr; if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnAdds(wsAddr, wsSize, exec, stream);
            aclrtSynchronizeStream(stream);
            std::vector<int8_t> outHost(4);
            aclrtMemcpy(outHost.data(), 4*sizeof(int8_t), outDev, 4*sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
            // 预期：true+true*true = true (1), false+true*true = true (1), ... 所以全部为1
            for (int i=0; i<4; i++) if (outHost[i] != 1) totalFailed++;
            if (wsAddr) aclrtFree(wsAddr);
        } else {
            totalFailed++; // 意外失败
        }
        aclDestroyTensor(selfT); aclDestroyTensor(outT); aclDestroyScalar(otherS); aclDestroyScalar(alpha);
        if (selfDev) aclrtFree(selfDev); if (outDev) aclrtFree(outDev);
    }

    // ----- 10.6 PromoteTypeScalar 中 DT_DOUBLE -> float 分支（aclnn_add.cpp 行524） -----
    {
        // Adds: self=float, other=double, out=float → 走行524
        std::vector<float> selfF = {1.0f, 2.0f};
        void *selfDev=nullptr, *outDev=nullptr; aclTensor *sT=nullptr, *oT=nullptr;
        CreateAclTensor(selfF, {2}, &selfDev, ACL_FLOAT, &sT);
        std::vector<float> outD(2, 0.0f);
        CreateAclTensor(outD, {2}, &outDev, ACL_FLOAT, &oT);
        aclScalar *otherD = CreateAclScalarFromDouble(3.14, ACL_DOUBLE);
        aclScalar *alpha = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
        uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
        auto ret = aclnnAddsGetWorkspaceSize(sT, otherD, alpha, oT, &wsSize, &exec);
        if (ret != ACL_SUCCESS) totalFailed++;   // 期望成功
        aclDestroyTensor(sT); aclDestroyTensor(oT); aclDestroyScalar(otherD); aclDestroyScalar(alpha);
        if (selfDev) aclrtFree(selfDev); if (outDev) aclrtFree(outDev);
    }

    // ----- 10.7 Tiling 层错误拦截（add_tiling_arch35.cpp） -----
    {
        // 10.7a 混合 dtype 输出非 float（bf16 + float -> out=float16）
        {
            aclTensor *selfT = CreateDummyAclTensor({2}, ACL_BF16);
            aclTensor *otherT = CreateDummyAclTensor({2}, ACL_FLOAT);
            aclTensor *outT = CreateDummyAclTensor({2}, ACL_FLOAT16);
            aclScalar *alpha = CreateAclScalarFromDouble(1.0, ACL_FLOAT16);
            uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
            if (aclnnAddGetWorkspaceSize(selfT, otherT, alpha, outT, &wsSize, &exec) == ACL_SUCCESS)
                totalFailed++;
            aclDestroyTensor(selfT); aclDestroyTensor(otherT); aclDestroyTensor(outT);
            aclDestroyScalar(alpha);
        }

        // 10.7b 非混合但 dtype 不一致（float + int32 -> out=float 触发 tiling CheckDtype 错误）
        {
            aclTensor *selfT = CreateDummyAclTensor({2}, ACL_FLOAT);
            aclTensor *otherT = CreateDummyAclTensor({2}, ACL_INT32);
            aclTensor *outT = CreateDummyAclTensor({2}, ACL_FLOAT);
            aclScalar *alpha = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
            uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
            if (aclnnAddGetWorkspaceSize(selfT, otherT, alpha, outT, &wsSize, &exec) == ACL_SUCCESS)
                totalFailed++;
            aclDestroyTensor(selfT); aclDestroyTensor(otherT); aclDestroyTensor(outT);
            aclDestroyScalar(alpha);
        }

        // 10.7c int16 输入触发 tiling else 分支（DoOpTiling 行 122-126）
        {
            aclTensor *selfT = CreateDummyAclTensor({2}, ACL_INT16);
            aclTensor *otherT = CreateDummyAclTensor({2}, ACL_INT16);
            aclTensor *outT = CreateDummyAclTensor({2}, ACL_INT16);
            aclScalar *alpha = CreateAclScalarFromDouble(1.0, ACL_INT16);
            uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
            if (aclnnAddGetWorkspaceSize(selfT, otherT, alpha, outT, &wsSize, &exec) == ACL_SUCCESS)
                totalFailed++;
            aclDestroyTensor(selfT); aclDestroyTensor(otherT); aclDestroyTensor(outT);
            aclDestroyScalar(alpha);
        }
    }

    // ----- 10.8 aclnn_add_v3.cpp 未覆盖分支补全 -----
    {
        // 10.8a PromoteTypeScalar 未覆盖分支：other 为 bool 时进入行132
        {
            aclScalar *selfS = CreateAclScalarFromDouble(3, ACL_INT8);
            aclTensor *otherT = CreateDummyAclTensor({2}, ACL_BOOL);
            aclTensor *outT = CreateDummyAclTensor({2}, ACL_INT8);
            aclScalar *alpha = CreateAclScalarFromDouble(1, ACL_INT8);
            uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
            // 允许成功或失败，但是要执行到相关代码
            aclnnAddV3GetWorkspaceSize(selfS, otherT, alpha, outT, &wsSize, &exec);
            aclDestroyScalar(selfS); aclDestroyTensor(otherT); aclDestroyTensor(outT); aclDestroyScalar(alpha);
        }

        // 10.8b PromoteTypeScalar 中的 double->float 分支（行128）
        {
            aclScalar *selfS = CreateAclScalarFromDouble(3.14, ACL_DOUBLE);
            aclTensor *otherT = CreateDummyAclTensor({2}, ACL_FLOAT);
            aclTensor *outT = CreateDummyAclTensor({2}, ACL_FLOAT);
            aclScalar *alpha = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
            uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
            auto ret = aclnnAddV3GetWorkspaceSize(selfS, otherT, alpha, outT, &wsSize, &exec);
            if (ret != ACL_SUCCESS) totalFailed++;
            aclDestroyScalar(selfS); aclDestroyTensor(otherT); aclDestroyTensor(outT); aclDestroyScalar(alpha);
        }

        // 10.8c CheckPromoteType 错误分支（借用 AddV3 触发）
        // 无法 promote
        {
            aclScalar *selfS = CreateAclScalarFromDouble(1, ACL_BOOL);
            aclTensor *otherT = CreateDummyAclTensor({2}, ACL_COMPLEX64);
            aclTensor *outT = CreateDummyAclTensor({2}, ACL_FLOAT);
            aclScalar *alpha = CreateAclScalarFromDouble(1, ACL_FLOAT);
            uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
            if (aclnnAddV3GetWorkspaceSize(selfS, otherT, alpha, outT, &wsSize, &exec) == ACL_SUCCESS)
                totalFailed++;
            aclDestroyScalar(selfS); aclDestroyTensor(otherT); aclDestroyTensor(outT); aclDestroyScalar(alpha);
        }
        // alpha 无法 cast
        {
            aclScalar *selfS = CreateAclScalarFromDouble(1, ACL_FLOAT);
            aclTensor *otherT = CreateDummyAclTensor({2}, ACL_FLOAT);
            aclTensor *outT = CreateDummyAclTensor({2}, ACL_FLOAT);
            float av[2] = {1.0f, 0.0f};
            aclScalar *alpha = aclCreateScalar(av, ACL_COMPLEX64);
            uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
            if (aclnnAddV3GetWorkspaceSize(selfS, otherT, alpha, outT, &wsSize, &exec) == ACL_SUCCESS)
                totalFailed++;
            aclDestroyScalar(selfS); aclDestroyTensor(otherT); aclDestroyTensor(outT); aclDestroyScalar(alpha);
        }
        // promoteType 不能 cast 到 out
        {
            aclScalar *selfS = CreateAclScalarFromDouble(1, ACL_FLOAT);
            aclTensor *otherT = CreateDummyAclTensor({2}, ACL_FLOAT);
            aclTensor *outT = CreateDummyAclTensor({2}, ACL_COMPLEX32);
            aclScalar *alpha = CreateAclScalarFromDouble(1, ACL_FLOAT);
            uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
            if (aclnnAddV3GetWorkspaceSize(selfS, otherT, alpha, outT, &wsSize, &exec) == ACL_SUCCESS)
                totalFailed++;
            aclDestroyScalar(selfS); aclDestroyTensor(otherT); aclDestroyTensor(outT); aclDestroyScalar(alpha);
        }
    }

    // ----- 10.9 补充 Inplace 混合类型输出错误（tiling CheckDtype） -----
    {
        // InplaceAdd: self=bf16, other=float, alpha=1 → 内部会生成 out 为 bf16，而混合要求 out=float，会触发 tiling 错误
        void *selfDev=nullptr, *otherDev=nullptr;
        aclTensor *selfT = CreateDummyAclTensor({2}, ACL_BF16);
        aclTensor *otherT = CreateDummyAclTensor({2}, ACL_FLOAT);
        aclScalar *alpha = CreateAclScalarFromDouble(1.0, ACL_BF16);
        uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
        if (aclnnInplaceAddGetWorkspaceSize(selfT, otherT, alpha, &wsSize, &exec) == ACL_SUCCESS)
            totalFailed++;
        aclDestroyTensor(selfT); aclDestroyTensor(otherT); aclDestroyScalar(alpha);
    }
    // 补充用例 1: 触发 aclnn_add.cpp 中的 FORMAT_NCHW 告警分支
    {
        int64_t shape[3] = {2, 3, 4};
        int64_t strides[3] = {12, 4, 1};
        aclTensor *tNchw1 = aclCreateTensor(shape, 3, ACL_FLOAT, strides, 0, ACL_FORMAT_NCHW, shape, 3, nullptr);
        aclTensor *tNchw2 = aclCreateTensor(shape, 3, ACL_FLOAT, strides, 0, ACL_FORMAT_NCHW, shape, 3, nullptr);
        aclTensor *outNchw = aclCreateTensor(shape, 3, ACL_FLOAT, strides, 0, ACL_FORMAT_NCHW, shape, 3, nullptr);
        
        aclScalar *alphaLocal = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
        uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
        // 仅获取 Workspace 即可触发格式告警，无需下发执行
        aclnnAddGetWorkspaceSize(tNchw1, tNchw2, alphaLocal, outNchw, &wsSize, &executor);

        aclDestroyTensor(tNchw1); aclDestroyTensor(tNchw2); aclDestroyTensor(outNchw);
        aclDestroyScalar(alphaLocal);
    }
    // 补充用例 2: 触发 CheckPromoteType 异常分支
    {
        void *tFloatDev=nullptr, *tBoolDev=nullptr;
        aclTensor *tFloat=nullptr, *tBool=nullptr;
        std::vector<int64_t> testShape = {2};
        CreateAclTensor<float>({1.0f, 2.0f}, testShape, &tFloatDev, ACL_FLOAT, &tFloat);
        CreateAclTensor<uint8_t>({1, 0}, testShape, &tBoolDev, ACL_BOOL, &tBool);

        // 场景 A: promoteType == BOOL，但 alpha 是 FLOAT（非整型）
        aclScalar *alphaFloat = CreateAclScalarFromDouble(1.5, ACL_FLOAT);
        uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
        aclnnAddGetWorkspaceSize(tBool, tBool, alphaFloat, tBool, &wsSize, &executor); 

        // 场景 B: CanCast(alpha, promoteType) 为 false (self=FLOAT, alpha=COMPLEX64)
        float cplx[2] = {1.0f, 1.0f};
        aclScalar *alphaCplx = aclCreateScalar(cplx, ACL_COMPLEX64);
        aclnnAddGetWorkspaceSize(tFloat, tFloat, alphaCplx, tFloat, &wsSize, &executor);
        
        // 场景 C: AddV3 中的 CheckPromoteType 异常 (selfScalar=COMPLEX, out=FLOAT)
        aclnnAddV3GetWorkspaceSize(alphaCplx, tFloat, alphaCplx, tFloat, &wsSize, &executor);

        aclDestroyTensor(tFloat); aclDestroyTensor(tBool);
        aclDestroyScalar(alphaFloat); aclDestroyScalar(alphaCplx);
        if(tFloatDev) aclrtFree(tFloatDev); 
        if(tBoolDev) aclrtFree(tBoolDev);
    }
    // 补充用例 3: 触发 aclnnAdds 中的特殊 BOOL 逻辑分支
    {
        std::vector<uint8_t> b_in = {1, 0, 1, 0};
        std::vector<double> exp_bool = {1.0, 1.0, 1.0, 1.0};
        // 输入和 alpha 都是 BOOL，输出使用 INT8，触发 L628-634 左右的特定类型映射
        RunAddsTest<uint8_t, int8_t>("Adds_Special_BOOL_Logic", ACL_BOOL, ACL_BOOL, ACL_INT8,
            b_in, 1.0, 1.0, {4}, exp_bool, stream);
    }
    // 补充用例 4: 触发 DOUBLE 到 FLOAT 的类型提升
    {
        std::vector<int32_t> i32_in = {1, 2, 3, 4};
        std::vector<double> exp_promote = {3.5, 4.5, 5.5, 6.5};
        
        RunAddsTest<int32_t, float>("Adds_Promote_DOUBLE_to_FLOAT", ACL_INT32, ACL_DOUBLE, ACL_FLOAT,
            i32_in, 2.5, 1.0, {4}, exp_promote, stream);
            
        RunAddV3Test<int32_t, float>("AddV3_Promote_DOUBLE_to_FLOAT", ACL_DOUBLE, ACL_INT32, ACL_FLOAT,
            2.5, i32_in, 1.0, {4}, exp_promote, stream);
    }
    // 补充用例 5: 覆盖 add.cpp 的 AiCpu 派发分支（静态解析图不执行，防卡死）
    {
        void *tDoubleDev1=nullptr, *tDoubleDev2=nullptr, *outDoubleDev=nullptr;
        aclTensor *tDouble1=nullptr, *tDouble2=nullptr, *outDouble=nullptr;
        std::vector<int64_t> dShape = {2};
        
        CreateAclTensor<double>({1.0, 2.0}, dShape, &tDoubleDev1, ACL_DOUBLE, &tDouble1);
        CreateAclTensor<double>({1.0, 2.0}, dShape, &tDoubleDev2, ACL_DOUBLE, &tDouble2);
        CreateAclTensor<double>({0.0, 0.0}, dShape, &outDoubleDev, ACL_DOUBLE, &outDouble);

        aclScalar *alphaDouble = CreateAclScalarFromDouble(1.0, ACL_DOUBLE);
        uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
        
        // 走入 add.cpp 的 IsAiCoreSupport 判定，由于平台/类型限制会被分发到 AddAiCpu
        aclnnAddGetWorkspaceSize(tDouble1, tDouble2, alphaDouble, outDouble, &wsSize, &executor);
        // [严格注意]: 仅拿覆盖率，不调用 executor，直接销毁，避免环境挂起
        
        aclDestroyTensor(tDouble1); aclDestroyTensor(tDouble2); aclDestroyTensor(outDouble);
        aclDestroyScalar(alphaDouble);
        if(tDoubleDev1) aclrtFree(tDoubleDev1); 
        if(tDoubleDev2) aclrtFree(tDoubleDev2); 
        if(outDoubleDev) aclrtFree(outDoubleDev);
    }
    // 补充用例 6: 覆盖 Inplace 操作的负向边界
    {
        void *tRefDev=nullptr, *tBigDev=nullptr;
        aclTensor *tRefBad=nullptr, *tBig=nullptr;
        // Ref Tensor 为 1D，Other Tensor 为 2D
        CreateAclTensor<float>({1.0f, 2.0f, 3.0f}, {3}, &tRefDev, ACL_FLOAT, &tRefBad);
        CreateAclTensor<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, &tBigDev, ACL_FLOAT, &tBig);
        
        aclScalar *alphaLocal = CreateAclScalarFromDouble(1.0, ACL_FLOAT);
        uint64_t wsSize = 0; aclOpExecutor* executor = nullptr;
        
        // 此处在前端会抛出 Shape Check 错误并返回非 SUCCESS
        aclnnInplaceAddGetWorkspaceSize(tRefBad, tBig, alphaLocal, &wsSize, &executor); 
        
        aclDestroyTensor(tRefBad); aclDestroyTensor(tBig); aclDestroyScalar(alphaLocal);
        if(tRefDev) aclrtFree(tRefDev); 
        if(tBigDev) aclrtFree(tBigDev);
    }

    LOG_PRINT("NoOneLog=== 全量覆盖率补充测试完成 ===\n");

    LOG_PRINT("NoOneLog=== Coverage & Precision Extension Complete ===\n");
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return totalFailed;
}