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

// Helper to get Shape Size
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (auto s : shape) size *= s;
    return size;
}

// Data Type conversion helpers
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

// Result verification
bool Compare(float actual, float expected, aclDataType dtype) {
    if (dtype == ACL_BOOL) return (actual != 0) == (expected != 0);
    float atol = 1e-3, rtol = 1e-3;
    if (dtype == ACL_FLOAT16 || dtype == ACL_BF16) { atol = 1e-2; rtol = 1e-2; }
    if (dtype == ACL_INT8 || dtype == ACL_UINT8 || dtype == ACL_INT32 || dtype == ACL_INT64) {
        return std::abs(actual - expected) < 0.5f;
    }
    return std::abs(actual - expected) <= (atol + rtol * std::abs(expected));
}

// Test Runner
bool RunTestAdd(const std::string& name, 
                const std::vector<int64_t>& shape1, const std::vector<float>& data1, aclDataType dtype1,
                const std::vector<int64_t>& shape2, const std::vector<float>& data2, aclDataType dtype2,
                float alphaVal, aclDataType dtypeAlpha,
                const std::vector<int64_t>& shapeOut, aclDataType dtypeOut,
                int apiVariant = 0) {
    
    LOG_PRINT("Testing %s ... ", name.c_str());

    void *d1 = nullptr, *d2 = nullptr, *dout = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *tout = nullptr;
    
    // Alpha creation
    int64_t i64Alpha = (int64_t)alphaVal;
    int32_t i32Alpha = (int32_t)alphaVal;
    int16_t i16Alpha = (int16_t)alphaVal;
    int8_t i8Alpha = (int8_t)alphaVal;
    uint8_t u8Alpha = (uint8_t)alphaVal;
    float f32Alpha = alphaVal;
    float16 f16Alpha = float_to_fp16(alphaVal);
    bfloat16 bf16Alpha = float_to_bf16(alphaVal);
    void* pAlpha = nullptr;
    if (dtypeAlpha == ACL_INT64) pAlpha = &i64Alpha;
    else if (dtypeAlpha == ACL_INT32) pAlpha = &i32Alpha;
    else if (dtypeAlpha == ACL_INT16) pAlpha = &i16Alpha;
    else if (dtypeAlpha == ACL_INT8) pAlpha = &i8Alpha;
    else if (dtypeAlpha == ACL_UINT8) pAlpha = &u8Alpha;
    else if (dtypeAlpha == ACL_FLOAT16) pAlpha = &f16Alpha;
    else if (dtypeAlpha == ACL_BF16) pAlpha = &bf16Alpha;
    else pAlpha = &f32Alpha;
    aclScalar *alpha = aclCreateScalar(pAlpha, dtypeAlpha);

    aclScalar *s2 = nullptr;

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
        return -1;
    };

    if (apiVariant != 4 && apiVariant != 5) fill_tensor(data1, dtype1, &d1, &t1, shape1);
    if (apiVariant == 0 || apiVariant == 2 || apiVariant == 4 || apiVariant == 5) fill_tensor(data2, dtype2, &d2, &t2, shape2);
    else {
        if (dtype2 == ACL_INT32) { int32_t v = (int32_t)data2[0]; s2 = aclCreateScalar(&v, dtype2); }
        else if (dtype2 == ACL_INT64) { int64_t v = (int64_t)data2[0]; s2 = aclCreateScalar(&v, dtype2); }
        else { float v = data2[0]; s2 = aclCreateScalar(&v, dtype2); }
    }

    if (apiVariant == 2 || apiVariant == 3) fill_tensor(data1, dtypeOut, &dout, &tout, shapeOut);
    else if (apiVariant == 5) fill_tensor(data2, dtypeOut, &dout, &tout, shapeOut);
    else {
        std::vector<float> outInit(GetShapeSize(shapeOut), 0);
        fill_tensor(outInit, dtypeOut, &dout, &tout, shapeOut);
    }

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    int ret = 0;
    void* wsAddr = nullptr;

    if (apiVariant == 0) ret = aclnnAddGetWorkspaceSize(t1, t2, alpha, tout, &wsSize, &exec);
    else if (apiVariant == 1) ret = aclnnAddsGetWorkspaceSize(t1, s2, alpha, tout, &wsSize, &exec);
    else if (apiVariant == 2) ret = aclnnInplaceAddGetWorkspaceSize(tout, t2, alpha, &wsSize, &exec);
    else if (apiVariant == 3) ret = aclnnInplaceAddsGetWorkspaceSize(tout, s2, alpha, &wsSize, &exec);
    else if (apiVariant == 4) { 
        aclScalar* s1 = nullptr;
        if (dtype1 == ACL_INT32) { int32_t v = (int32_t)data1[0]; s1 = aclCreateScalar(&v, dtype1); }
        else { float v = data1[0]; s1 = aclCreateScalar(&v, dtype1); }
        ret = aclnnAddV3GetWorkspaceSize(s1, t2, alpha, tout, &wsSize, &exec); aclDestroyScalar(s1); 
    }
    else if (apiVariant == 5) { 
        aclScalar* s1 = nullptr;
        if (dtype1 == ACL_INT32) { int32_t v = (int32_t)data1[0]; s1 = aclCreateScalar(&v, dtype1); }
        else { float v = data1[0]; s1 = aclCreateScalar(&v, dtype1); }
        ret = aclnnInplaceAddV3GetWorkspaceSize(s1, tout, alpha, &wsSize, &exec); aclDestroyScalar(s1); 
    }

    if (ret != 0) { LOG_PRINT("[FAIL] GetWorkspaceSize error %d\n", ret); goto cleanup; }

    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (apiVariant == 0) ret = aclnnAdd(wsAddr, wsSize, exec, stream);
    else if (apiVariant == 1) ret = aclnnAdds(wsAddr, wsSize, exec, stream);
    else if (apiVariant == 2) ret = aclnnInplaceAdd(wsAddr, wsSize, exec, stream);
    else if (apiVariant == 3) ret = aclnnInplaceAdds(wsAddr, wsSize, exec, stream);
    else if (apiVariant == 4) ret = aclnnAddV3(wsAddr, wsSize, exec, stream);
    else if (apiVariant == 5) ret = aclnnInplaceAddV3(wsAddr, wsSize, exec, stream);

    aclrtSynchronizeStream(stream);

    if (ret == 0) {
        int64_t szOut = GetShapeSize(shapeOut);
        void* hostRes = malloc(szOut * 8);
        aclrtMemcpy(hostRes, szOut * 8, dout, szOut * (dtypeOut == ACL_INT64 ? 8 : 4), ACL_MEMCPY_DEVICE_TO_HOST);
        
        std::vector<float> res(szOut);
        for (int i=0; i<szOut; ++i) {
            if (dtypeOut == ACL_FLOAT) res[i] = ((float*)hostRes)[i];
            else if (dtypeOut == ACL_FLOAT16) res[i] = fp16_to_float(((float16*)hostRes)[i]);
            else if (dtypeOut == ACL_BF16) res[i] = bf16_to_float(((bfloat16*)hostRes)[i]);
            else if (dtypeOut == ACL_INT32) res[i] = (float)((int32_t*)hostRes)[i];
            else if (dtypeOut == ACL_INT8) res[i] = (float)((int8_t*)hostRes)[i];
            else if (dtypeOut == ACL_UINT8) res[i] = (float)((uint8_t*)hostRes)[i];
            else if (dtypeOut == ACL_INT64) res[i] = (float)((int64_t*)hostRes)[i];
            else if (dtypeOut == ACL_BOOL) res[i] = (float)((uint8_t*)hostRes)[i];
        }

        bool pass = true;
        for (int i=0; i<szOut; ++i) {
            float v1 = 0, v2 = 0;
            if (apiVariant == 4 || apiVariant == 5) {
                v1 = data1[0]; v2 = data2[i % data2.size()];
            } else if (apiVariant == 1 || apiVariant == 3) {
                v1 = data1[i % data1.size()]; v2 = data2[0];
            } else {
                int r = i / (int)shapeOut.back(), c = i % (int)shapeOut.back();
                auto get_val = [&](const std::vector<float>& d, const std::vector<int64_t>& sh) {
                    if (sh.size() == 1 && sh[0] == 1) return d[0];
                    int rr = (sh.size() > 1 && sh[sh.size()-2] > 1) ? r : 0;
                    int cc = (sh.back() > 1) ? c : 0;
                    return d[rr * (int)sh.back() + cc];
                };
                v1 = get_val(data1, shape1); v2 = get_val(data2, shape2);
            }
            float expected = v1 + alphaVal * v2;
            if (!Compare(res[i], expected, dtypeOut)) {
                LOG_PRINT("[FAIL] at index %d: actual %f, expected %f (v1=%f, v2=%f, alpha=%f)\n", i, res[i], expected, v1, v2, alphaVal);
                pass = false; break;
            }
        }
        if (pass) LOG_PRINT("[PASS]\n");
        free(hostRes);
        if (!pass) { ret = -1; }
    } else {
        LOG_PRINT("[FAIL] Execute error %d\n", ret);
    }
    if (wsAddr) aclrtFree(wsAddr);

cleanup:
    if (d1) aclrtFree(d1); if (d2) aclrtFree(d2); if (dout) aclrtFree(dout);
    aclDestroyTensor(t1); aclDestroyTensor(t2); aclDestroyTensor(tout);
    aclDestroyScalar(alpha); if (s2) aclDestroyScalar(s2);
    return (ret == 0);
}

int main() {
    if (Init() != 0) return -1;
    bool all_pass = true;
    all_pass &= RunTestAdd("Add_FP32", {2, 4}, {1,2,3,4,5,6,7,8}, ACL_FLOAT, {2, 4}, {1,1,1,1,1,1,1,1}, ACL_FLOAT, 1.0f, ACL_FLOAT, {2, 4}, ACL_FLOAT, 0);
    all_pass &= RunTestAdd("Add_INT32", {2, 4}, {1,2,3,4,5,6,7,8}, ACL_INT32, {2, 4}, {10,10,10,10,10,10,10,10}, ACL_INT32, 1.0f, ACL_FLOAT, {2, 4}, ACL_INT32, 0);
    all_pass &= RunTestAdd("Add_Alpha", {2, 2}, {1,1,1,1}, ACL_FLOAT, {2, 2}, {2,2,2,2}, ACL_FLOAT, 1.5f, ACL_FLOAT, {2, 2}, ACL_FLOAT, 0);
    all_pass &= RunTestAdd("Add_FP16", {2, 2}, {1,2,3,4}, ACL_FLOAT16, {2, 2}, {1,1,1,1}, ACL_FLOAT16, 1.0f, ACL_FLOAT, {2, 2}, ACL_FLOAT16, 0);
    all_pass &= RunTestAdd("Add_BF16", {2, 2}, {1,2,3,4}, ACL_BF16, {2, 2}, {1,1,1,1}, ACL_BF16, 1.0f, ACL_FLOAT, {2, 2}, ACL_BF16, 0);
    all_pass &= RunTestAdd("Add_INT8", {2, 2}, {1,2,3,4}, ACL_INT8, {2, 2}, {1,1,1,1}, ACL_INT8, 1.0f, ACL_FLOAT, {2, 2}, ACL_INT8, 0);
    all_pass &= RunTestAdd("Add_UINT8", {2, 2}, {1,2,3,4}, ACL_UINT8, {2, 2}, {1,1,1,1}, ACL_UINT8, 1.0f, ACL_FLOAT, {2, 2}, ACL_UINT8, 0);
    all_pass &= RunTestAdd("Add_INT64", {2, 2}, {1,2,3,4}, ACL_INT64, {2, 2}, {1,1,1,1}, ACL_INT64, 1.0f, ACL_FLOAT, {2, 2}, ACL_INT64, 0);
    all_pass &= RunTestAdd("Add_BOOL", {2, 2}, {0,1,0,1}, ACL_BOOL, {2, 2}, {0,0,1,1}, ACL_BOOL, 1.0f, ACL_FLOAT, {2, 2}, ACL_BOOL, 0);
    all_pass &= RunTestAdd("Add_Broadcast_1", {2, 4}, {1,1,1,1,2,2,2,2}, ACL_FLOAT, {1, 4}, {1,2,3,4}, ACL_FLOAT, 1.0f, ACL_FLOAT, {2, 4}, ACL_FLOAT, 0);
    all_pass &= RunTestAdd("Add_Broadcast_2", {2, 4}, {1,2,3,4,5,6,7,8}, ACL_FLOAT, {2, 1}, {10,20}, ACL_FLOAT, 1.0f, ACL_FLOAT, {2, 4}, ACL_FLOAT, 0);
    all_pass &= RunTestAdd("Add_Mix_FP16_FP32", {2, 2}, {1,2,3,4}, ACL_FLOAT16, {2, 2}, {10,10,10,10}, ACL_FLOAT, 1.0f, ACL_FLOAT, {2, 2}, ACL_FLOAT, 0);
    all_pass &= RunTestAdd("Adds_FP32", {2, 2}, {1,2,3,4}, ACL_FLOAT, {1}, {10}, ACL_FLOAT, 1.0f, ACL_FLOAT, {2, 2}, ACL_FLOAT, 1);
    all_pass &= RunTestAdd("InplaceAdd_FP32", {2, 2}, {1,2,3,4}, ACL_FLOAT, {2, 2}, {10,10,10,10}, ACL_FLOAT, 1.0f, ACL_FLOAT, {2, 2}, ACL_FLOAT, 2);
    all_pass &= RunTestAdd("InplaceAdds_FP32", {2, 2}, {1,2,3,4}, ACL_FLOAT, {1}, {10}, ACL_FLOAT, 1.0f, ACL_FLOAT, {2, 2}, ACL_FLOAT, 3);
    all_pass &= RunTestAdd("AddV3_FP32", {1}, {100}, ACL_FLOAT, {2, 2}, {1,2,3,4}, ACL_FLOAT, 1.0f, ACL_FLOAT, {2, 2}, ACL_FLOAT, 4);
    all_pass &= RunTestAdd("InplaceAddV3_FP32", {1}, {100}, ACL_FLOAT, {2, 2}, {1,2,3,4}, ACL_FLOAT, 1.0f, ACL_FLOAT, {2, 2}, ACL_FLOAT, 5);
    all_pass &= RunTestAdd("Add_Alpha_Neg", {2, 2}, {10,20,30,40}, ACL_FLOAT, {2, 2}, {1,2,3,4}, ACL_FLOAT, -1.0f, ACL_FLOAT, {2, 2}, ACL_FLOAT, 0);
    Deinit();
    if (all_pass) { LOG_PRINT("All tests passed!\n"); return 0; }
    LOG_PRINT("Some tests failed.\n"); return -1;
}
