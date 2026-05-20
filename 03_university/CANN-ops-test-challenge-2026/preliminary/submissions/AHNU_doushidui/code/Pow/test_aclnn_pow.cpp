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
#include <cstring>
#include <algorithm>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"
#include "aclnnop/aclnn_exp2.h"

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

// Helper to get Shape Size
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (auto s : shape) size *= s;
    return size;
}

// Data Type helpers
typedef uint16_t bfloat16;
bfloat16 float_to_bf16(float f) {
    uint32_t i;
    std::memcpy(&i, &f, sizeof(i));
    return static_cast<bfloat16>(i >> 16);
}
float bf16_to_float(bfloat16 b) {
    uint32_t i = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &i, sizeof(i));
    return f;
}

typedef uint16_t float16;
float16 float_to_fp16(float f) {
    if (f == 0.0f) return 0;
    uint32_t i;
    std::memcpy(&i, &f, sizeof(i));
    uint32_t s = (i >> 16) & 0x8000;
    int32_t e = ((i >> 23) & 0xFF) - 127 + 15;
    uint32_t m = (i >> 13) & 0x03FF;
    if (e <= 0) return s;
    if (e >= 31) return s | 0x7C00;
    return s | (e << 10) | m;
}
float fp16_to_float(float16 h) {
    uint32_t s = (h & 0x8000) << 16;
    uint32_t e = (h & 0x7C00) >> 10;
    uint32_t m = (h & 0x03FF) << 13;
    if (e == 0) return s;
    if (e == 31) return s | 0x7F800000;
    e = e - 15 + 127;
    uint32_t i = s | (e << 23) | m;
    float f;
    std::memcpy(&f, &i, sizeof(i));
    return f;
}

size_t GetDtypeSize(aclDataType dt) {
    switch (dt) {
        case ACL_FLOAT: return 4;
        case ACL_FLOAT16: return 2;
        case ACL_BF16: return 2;
        case ACL_INT32: return 4;
        case ACL_INT16: return 2;
        case ACL_INT8: return 1;
        case ACL_UINT8: return 1;
        case ACL_INT64: return 8;
        case ACL_BOOL: return 1;
        case ACL_DOUBLE: return 8;
        default: return 4;
    }
}

// Global state
aclrtStream stream;
int32_t deviceId = 0;

int Init() {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}

void Deinit() {
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                   aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = (int)shape.size() - 2; i >= 0; i--) strides[i] = shape[i + 1] * strides[i + 1];
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

bool Compare(float actual, float expected, aclDataType dtype) {
    if (std::isnan(actual) && std::isnan(expected)) return true;
    if (std::isinf(actual) && std::isinf(expected)) return (actual > 0) == (expected > 0);
    if (dtype == ACL_BOOL) return (actual != 0) == (expected != 0);
    float atol = 1e-2, rtol = 1e-2;
    if (dtype == ACL_INT8 || dtype == ACL_UINT8 || dtype == ACL_INT32 || dtype == ACL_INT64 || dtype == ACL_INT16) {
        return std::abs(actual - expected) < 0.5f;
    }
    return std::abs(actual - expected) <= (atol + rtol * std::abs(expected));
}

// Test Runner
bool RunTestPow(const std::string& name, 
                const std::vector<int64_t>& shape1, const std::vector<float>& data1, aclDataType dtype1,
                const std::vector<int64_t>& shape2, const std::vector<float>& data2, aclDataType dtype2,
                const std::vector<int64_t>& shapeOut, aclDataType dtypeOut,
                int apiVariant = 0) {
    
    LOG_PRINT("Testing %s ... ", name.c_str());

    void *d1 = nullptr, *d2 = nullptr, *dout = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    aclScalar *s1 = nullptr, *s2 = nullptr;

    auto fill_tensor = [&](const std::vector<float>& src, aclDataType dt, void** dev, aclTensor** ten, const std::vector<int64_t>& sh) {
        int64_t sz = GetShapeSize(sh);
        if (dt == ACL_FLOAT) { std::vector<float> d(src); return CreateAclTensor(d, sh, dev, dt, ten); }
        if (dt == ACL_FLOAT16) { std::vector<float16> d(sz); for(int i=0;i<sz;++i) d[i]=float_to_fp16(src[i]); return CreateAclTensor(d, sh, dev, dt, ten); }
        if (dt == ACL_BF16) { std::vector<bfloat16> d(sz); for(int i=0;i<sz;++i) d[i]=float_to_bf16(src[i]); return CreateAclTensor(d, sh, dev, dt, ten); }
        if (dt == ACL_INT32) { std::vector<int32_t> d(sz); for(int i=0;i<sz;++i) d[i]=(int32_t)src[i]; return CreateAclTensor(d, sh, dev, dt, ten); }
        if (dt == ACL_INT16) { std::vector<int16_t> d(sz); for(int i=0;i<sz;++i) d[i]=(int16_t)src[i]; return CreateAclTensor(d, sh, dev, dt, ten); }
        if (dt == ACL_INT8) { std::vector<int8_t> d(sz); for(int i=0;i<sz;++i) d[i]=(int8_t)src[i]; return CreateAclTensor(d, sh, dev, dt, ten); }
        if (dt == ACL_UINT8) { std::vector<uint8_t> d(sz); for(int i=0;i<sz;++i) d[i]=(uint8_t)src[i]; return CreateAclTensor(d, sh, dev, dt, ten); }
        if (dt == ACL_INT64) { std::vector<int64_t> d(sz); for(int i=0;i<sz;++i) d[i]=(int64_t)src[i]; return CreateAclTensor(d, sh, dev, dt, ten); }
        if (dt == ACL_BOOL) { std::vector<uint8_t> d(sz); for(int i=0;i<sz;++i) d[i]=(src[i] != 0); return CreateAclTensor(d, sh, dev, dt, ten); }
        if (dt == ACL_DOUBLE) { std::vector<double> d(sz); for(int i=0;i<sz;++i) d[i]=(double)src[i]; return CreateAclTensor(d, sh, dev, dt, ten); }
        return -1;
    };

    auto create_scalar = [&](float val, aclDataType dt) {
        if (dt == ACL_INT32) { int32_t v = (int32_t)val; return aclCreateScalar(&v, dt); }
        if (dt == ACL_INT64) { int64_t v = (int64_t)val; return aclCreateScalar(&v, dt); }
        if (dt == ACL_FLOAT16) { float16 v = float_to_fp16(val); return aclCreateScalar(&v, dt); }
        if (dt == ACL_BF16) { bfloat16 v = float_to_bf16(val); return aclCreateScalar(&v, dt); }
        if (dt == ACL_DOUBLE) { double v = (double)val; return aclCreateScalar(&v, dt); }
        float fv = val; return aclCreateScalar(&fv, dt);
    };

    if (apiVariant != 2) fill_tensor(data1, dtype1, &d1, &t1, shape1);
    else s1 = create_scalar(data1[0], dtype1);

    if (apiVariant == 0 || apiVariant == 1) s2 = create_scalar(data2[0], dtype2);
    else if (apiVariant == 2 || apiVariant == 3 || apiVariant == 4) fill_tensor(data2, dtype2, &d2, &t2, shape2);

    if (apiVariant == 1 || apiVariant == 4 || apiVariant == 6) { tout = t1; dout = d1; }
    else {
        std::vector<float> outInit(GetShapeSize(shapeOut), 0);
        fill_tensor(outInit, dtypeOut, &dout, &tout, shapeOut);
    }

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    int ret = 0;
    void* wsAddr = nullptr;

    if (apiVariant == 0) ret = aclnnPowTensorScalarGetWorkspaceSize(t1, s2, tout, &wsSize, &exec);
    else if (apiVariant == 1) ret = aclnnInplacePowTensorScalarGetWorkspaceSize(t1, s2, &wsSize, &exec);
    else if (apiVariant == 2) ret = aclnnPowScalarTensorGetWorkspaceSize(s1, t2, tout, &wsSize, &exec);
    else if (apiVariant == 3) ret = aclnnPowTensorTensorGetWorkspaceSize(t1, t2, tout, &wsSize, &exec);
    else if (apiVariant == 4) ret = aclnnInplacePowTensorTensorGetWorkspaceSize(t1, t2, &wsSize, &exec);
    else if (apiVariant == 5) ret = aclnnExp2GetWorkspaceSize(t1, tout, &wsSize, &exec);
    else if (apiVariant == 6) ret = aclnnInplaceExp2GetWorkspaceSize(t1, &wsSize, &exec);

    if (ret != 0) { LOG_PRINT("[FAIL] GetWorkspaceSize error %d\n", ret); goto cleanup; }
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (apiVariant == 0) ret = aclnnPowTensorScalar(wsAddr, wsSize, exec, stream);
    else if (apiVariant == 1) ret = aclnnInplacePowTensorScalar(wsAddr, wsSize, exec, stream);
    else if (apiVariant == 2) ret = aclnnPowScalarTensor(wsAddr, wsSize, exec, stream);
    else if (apiVariant == 3) ret = aclnnPowTensorTensor(wsAddr, wsSize, exec, stream);
    else if (apiVariant == 4) ret = aclnnInplacePowTensorTensor(wsAddr, wsSize, exec, stream);
    else if (apiVariant == 5) ret = aclnnExp2(wsAddr, wsSize, exec, stream);
    else if (apiVariant == 6) ret = aclnnInplaceExp2(wsAddr, wsSize, exec, stream);

    aclrtSynchronizeStream(stream);

    if (ret == 0) {
        int64_t szOut = GetShapeSize(shapeOut);
        size_t dtSize = GetDtypeSize(dtypeOut);
        void* hostRes = malloc(szOut * 8);
        aclrtMemcpy(hostRes, szOut * dtSize, dout, szOut * dtSize, ACL_MEMCPY_DEVICE_TO_HOST);
        std::vector<float> res(szOut);
        for (int i=0; i<szOut; ++i) {
            if (dtypeOut == ACL_FLOAT) res[i] = ((float*)hostRes)[i];
            else if (dtypeOut == ACL_FLOAT16) res[i] = fp16_to_float(((float16*)hostRes)[i]);
            else if (dtypeOut == ACL_BF16) res[i] = bf16_to_float(((bfloat16*)hostRes)[i]);
            else if (dtypeOut == ACL_INT32) res[i] = (float)((int32_t*)hostRes)[i];
            else if (dtypeOut == ACL_INT16) res[i] = (float)((int16_t*)hostRes)[i];
            else if (dtypeOut == ACL_INT8) res[i] = (float)((int8_t*)hostRes)[i];
            else if (dtypeOut == ACL_UINT8) res[i] = (float)((uint8_t*)hostRes)[i];
            else if (dtypeOut == ACL_INT64) res[i] = (float)((int64_t*)hostRes)[i];
            else if (dtypeOut == ACL_BOOL) res[i] = (float)((uint8_t*)hostRes)[i];
            else if (dtypeOut == ACL_DOUBLE) res[i] = (float)((double*)hostRes)[i];
        }
        bool pass = true;
        for (int i=0; i<szOut; ++i) {
            double v1 = 0, v2 = 0;
            if (apiVariant == 5 || apiVariant == 6) { v1 = 2.0; v2 = data1[i % data1.size()]; }
            else if (apiVariant == 0 || apiVariant == 1) { v1 = data1[i % data1.size()]; v2 = data2[0]; }
            else if (apiVariant == 2) { v1 = data1[0]; v2 = data2[i % data2.size()]; }
            else {
                // Correct 2D broadcast logic for {2,1} and {1,2}
                int r = i / shapeOut.back(), c = i % shapeOut.back();
                auto get_v = [&](const std::vector<float>& d, const std::vector<int64_t>& sh) {
                    if (sh.size() == 1) return (double)d[i % sh[0]];
                    int rr = (sh[0] == 1) ? 0 : r;
                    int cc = (sh[1] == 1) ? 0 : c;
                    return (double)d[rr * sh[1] + cc];
                };
                v1 = get_v(data1, shape1); v2 = get_v(data2, shape2);
            }
            double expected = std::pow(v1, v2);
            if (!Compare(res[i], (float)expected, dtypeOut)) {
                LOG_PRINT("[FAIL] at index %d: actual %f, expected %f (v1=%f, v2=%f)\n", i, res[i], (float)expected, v1, v2);
                pass = false; break;
            }
        }
        if (pass) LOG_PRINT("[PASS]\n");
        free(hostRes);
        if (!pass) ret = -1;
    } else { LOG_PRINT("[FAIL] Execute error %d\n", ret); }

cleanup:
    if (wsAddr) aclrtFree(wsAddr);
    if (d1) aclrtFree(d1); if (d2) aclrtFree(d2); if (dout && dout != d1) aclrtFree(dout);
    if (t1) aclDestroyTensor(t1); if (t2) aclDestroyTensor(t2); if (tout && tout != t1) aclDestroyTensor(tout);
    if (s1) aclDestroyScalar(s1); if (s2) aclDestroyScalar(s2);
    return (ret == 0);
}

int main() {
    if (Init() != 0) return -1;
    bool all_pass = true;
    // Basic FLOAT cases
    all_pass &= RunTestPow("PowTS_FP32", {4}, {0, 1, 2, 3}, ACL_FLOAT, {1}, {4.0f}, ACL_FLOAT, {4}, ACL_FLOAT, 0);
    all_pass &= RunTestPow("PowTS_SQUARE", {2}, {2, 4}, ACL_FLOAT, {1}, {2.0f}, ACL_FLOAT, {2}, ACL_FLOAT, 0);
    all_pass &= RunTestPow("PowTS_SQRT", {2}, {4, 16}, ACL_FLOAT, {1}, {0.5f}, ACL_FLOAT, {2}, ACL_FLOAT, 0);
    all_pass &= RunTestPow("PowTS_CUBE", {2}, {2, 3}, ACL_FLOAT, {1}, {3.0f}, ACL_FLOAT, {2}, ACL_FLOAT, 0);
    all_pass &= RunTestPow("PowTS_NEG_ONE", {2}, {2, 4}, ACL_FLOAT, {1}, {-1.0f}, ACL_FLOAT, {2}, ACL_FLOAT, 0);
    all_pass &= RunTestPow("PowTS_NEG_SQRT", {2}, {4, 16}, ACL_FLOAT, {1}, {-0.5f}, ACL_FLOAT, {2}, ACL_FLOAT, 0);
    all_pass &= RunTestPow("PowTS_NEG_SQUARE", {2}, {2, 4}, ACL_FLOAT, {1}, {-2.0f}, ACL_FLOAT, {2}, ACL_FLOAT, 0);
    
    // Dtypes
    all_pass &= RunTestPow("PowTS_INT32", {2}, {2, 3}, ACL_INT32, {1}, {2.0f}, ACL_FLOAT, {2}, ACL_FLOAT, 0);
    all_pass &= RunTestPow("PowTS_INT16", {2}, {2, 3}, ACL_INT16, {1}, {2.0f}, ACL_FLOAT, {2}, ACL_FLOAT, 0);
    all_pass &= RunTestPow("PowTS_INT8", {2}, {2, 3}, ACL_INT8, {1}, {2.0f}, ACL_FLOAT, {2}, ACL_FLOAT, 0);
    all_pass &= RunTestPow("PowTS_UINT8", {2}, {2, 3}, ACL_UINT8, {1}, {2.0f}, ACL_FLOAT, {2}, ACL_FLOAT, 0);
    all_pass &= RunTestPow("PowTS_FP16", {2}, {2, 3}, ACL_FLOAT16, {1}, {2.0f}, ACL_FLOAT, {2}, ACL_FLOAT, 0);
    all_pass &= RunTestPow("PowTS_BF16", {2}, {2, 3}, ACL_BF16, {1}, {2.0f}, ACL_FLOAT, {2}, ACL_FLOAT, 0);
    all_pass &= RunTestPow("PowTS_BOOL", {2}, {0, 1}, ACL_BOOL, {1}, {2.0f}, ACL_FLOAT, {2}, ACL_FLOAT, 0);
    all_pass &= RunTestPow("PowTS_DOUBLE", {2}, {2, 3}, ACL_DOUBLE, {1}, {2.0f}, ACL_DOUBLE, {2}, ACL_DOUBLE, 0);

    // API Variants
    all_pass &= RunTestPow("PowTS_Inplace", {2}, {2, 3}, ACL_FLOAT, {1}, {2.0f}, ACL_FLOAT, {2}, ACL_FLOAT, 1);
    all_pass &= RunTestPow("PowST_FP32", {1}, {2.0f}, ACL_FLOAT, {4}, {0, 1, 2, 3}, ACL_FLOAT, {4}, ACL_FLOAT, 2);
    all_pass &= RunTestPow("PowST_Base1", {1}, {1.0f}, ACL_FLOAT, {2}, {2, 3}, ACL_FLOAT, {2}, ACL_FLOAT, 2);
    all_pass &= RunTestPow("PowTT_FP32", {4}, {2, 2, 2, 2}, ACL_FLOAT, {4}, {0, 1, 2, 3}, ACL_FLOAT, {4}, ACL_FLOAT, 3);
    all_pass &= RunTestPow("PowTT_Inplace", {2}, {2, 3}, ACL_FLOAT, {2}, {2, 2}, ACL_FLOAT, {2}, ACL_FLOAT, 4);
    all_pass &= RunTestPow("Exp2_FP32", {4}, {0, 1, 2, 3}, ACL_FLOAT, {1}, {0}, ACL_FLOAT, {4}, ACL_FLOAT, 5);
    all_pass &= RunTestPow("Exp2_Inplace", {2}, {1, 2}, ACL_FLOAT, {1}, {0}, ACL_FLOAT, {2}, ACL_FLOAT, 6);
    all_pass &= RunTestPow("Exp2_INT32", {2}, {1, 2}, ACL_INT32, {1}, {0}, ACL_FLOAT, {2}, ACL_FLOAT, 5);

    // More T-T Dtypes
    all_pass &= RunTestPow("PowTT_INT32", {2}, {2, 3}, ACL_INT32, {2}, {2, 2}, ACL_INT32, {2}, ACL_INT32, 3);
    all_pass &= RunTestPow("PowTT_FP16", {2}, {2, 3}, ACL_FLOAT16, {2}, {2, 2}, ACL_FLOAT16, {2}, ACL_FLOAT16, 3);
    all_pass &= RunTestPow("PowTT_BF16", {2}, {2, 3}, ACL_BF16, {2}, {2, 2}, ACL_BF16, {2}, ACL_BF16, 3);
    all_pass &= RunTestPow("PowTT_INT16", {2}, {2, 3}, ACL_INT16, {2}, {2, 2}, ACL_INT16, {2}, ACL_INT16, 3);
    all_pass &= RunTestPow("PowTT_INT8", {2}, {2, 3}, ACL_INT8, {2}, {2, 2}, ACL_INT8, {2}, ACL_INT8, 3);
    all_pass &= RunTestPow("PowTT_UINT8", {2}, {2, 3}, ACL_UINT8, {2}, {2, 2}, ACL_UINT8, {2}, ACL_UINT8, 3);

    // Edge Cases
    all_pass &= RunTestPow("PowTT_ZeroZero", {1}, {0}, ACL_FLOAT, {1}, {0}, ACL_FLOAT, {1}, ACL_FLOAT, 3);
    all_pass &= RunTestPow("PowTS_NegBaseIntExp", {1}, {-2.0f}, ACL_FLOAT, {1}, {3.0f}, ACL_FLOAT, {1}, ACL_FLOAT, 0);
    all_pass &= RunTestPow("PowTS_NegBaseFloatExp", {1}, {-2.0f}, ACL_FLOAT, {1}, {0.5f}, ACL_FLOAT, {1}, ACL_FLOAT, 0);
    all_pass &= RunTestPow("PowTT_Broadcast", {2, 1}, {2, 3}, ACL_FLOAT, {1, 2}, {2, 3}, ACL_FLOAT, {2, 2}, ACL_FLOAT, 3);

    Deinit();
    if (all_pass) { LOG_PRINT("All Pow tests passed!\n"); return 0; }
    LOG_PRINT("Some Pow tests failed.\n"); return -1;
}
