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
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

int failCount = 0;

void Check(bool pass, const char* testName) {
    if (pass) {
        printf("[PASS] %s\n", testName);
    } else {
        printf("[FAIL] %s\n", testName);
        failCount++;
    }
}

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) { shapeSize *= i; }
    return shapeSize;
}

aclTensor* CreateTensor(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, bool contiguous = true) {
    auto size = GetShapeSize(shape) * 8; 
    if (size == 0) size = 8;
    aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    
    std::vector<int64_t> strides(shape.size(), 1);
    if (contiguous) {
        for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    } else {
        // 故意制造非连续步长触发 IsAddSupportNonContiguous
        for (size_t i = 0; i < shape.size(); i++) strides[i] = 2;
    }
    return aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
}

void Test_Add_Standard_And_Axpy_Branches(aclrtStream stream) {
    // 1. 【修改点】结构体增加 dtype2，以支持混合精度覆盖
    struct TestCase {
        const char* name;
        aclDataType dtype1; // 输入1类型
        aclDataType dtype2; // 输入2类型
        float alpha;
        std::vector<int64_t> s_shape, o_shape;
    };

    // 2. 【修改点】对齐所有用例的参数个数（均为6个）
    TestCase cases[] = {
        // 原有用例对齐
        {"Add_FP32_Alpha1", ACL_FLOAT, ACL_FLOAT, 1.0f, {2, 2}, {2, 2}},
        {"Add_FP32_Axpy", ACL_FLOAT, ACL_FLOAT, 2.0f, {2, 2}, {2, 2}},
        {"Add_INT8_AxpyV2", ACL_INT8, ACL_INT8, 2.0f, {2, 2}, {2, 2}},
        {"Add_INT64_MulFallback", ACL_INT64, ACL_INT64, 2.0f, {2, 2}, {2, 2}},
        {"Add_Broadcast", ACL_FLOAT, ACL_FLOAT, 1.0f, {2, 2}, {2}},
        
        // 极限覆盖率用例
        {"FP32_Axpy", ACL_FLOAT, ACL_FLOAT, 3.14f, {8, 8}, {8, 8}},
        {"INT8_AxpyV2", ACL_INT8, ACL_INT8, 2.0f, {16}, {16}},
        {"BF16_Mix", ACL_BF16, ACL_FLOAT, 1.0f, {4, 4}, {4, 4}},         // 触发混合精度判断
        {"BOOL_Logic", ACL_BOOL, ACL_BOOL, 1.0f, {2, 2}, {2, 2}},        // 触发逻辑类型分支
        {"INT32_Std", ACL_INT32, ACL_INT32, 1.0f, {32}, {32}},           // 触发整型路径
        {"Large_Tiling", ACL_FLOAT, ACL_FLOAT, 1.0f, {1024}, {1024}},    // 触发多核并行Tiling
        {"Zero_Dim", ACL_FLOAT, ACL_FLOAT, 1.0f, {1, 0, 1}, {1, 0, 1}},  // 触发空Tensor异常处理
        {"Broadcast_Complex", ACL_FLOAT, ACL_FLOAT, 1.5f, {2, 1, 4}, {4}} // 触发复杂维度广播
    };

    for (auto& c : cases) {
        void *sAddr = nullptr, *oAddr = nullptr, *outAddr = nullptr, *wsAddr = nullptr;
        
        // 3. 【修改点】分别使用 dtype1 和 dtype2 创建 Tensor
        aclTensor *self = CreateTensor(c.s_shape, &sAddr, c.dtype1);
        aclTensor *other = CreateTensor(c.o_shape, &oAddr, c.dtype2);
        aclTensor *out = CreateTensor(c.s_shape, &outAddr, c.dtype1);
        aclScalar *alpha = aclCreateScalar(&c.alpha, ACL_FLOAT);

        uint64_t wsSize = 0;
        aclOpExecutor* exec = nullptr;
        auto ret1 = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
        
        if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        auto ret2 = aclnnAdd(wsAddr, wsSize, exec, stream);
        aclrtSynchronizeStream(stream);

        Check(ret1 == ACL_SUCCESS && ret2 == ACL_SUCCESS, c.name);

        aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out); aclDestroyScalar(alpha);
        aclrtFree(sAddr); aclrtFree(oAddr); aclrtFree(outAddr); if (wsSize > 0) aclrtFree(wsAddr);
    }
}

void Test_MixDtype_And_NonContiguous(aclrtStream stream) {
    // 覆盖: 混合精度 (FP16+FP32), 非连续Tensor
    void *sAddr = nullptr, *oAddr = nullptr, *outAddr = nullptr, *wsAddr = nullptr;
    std::vector<int64_t> shape = {4};
    
    aclTensor *self = CreateTensor(shape, &sAddr, ACL_FLOAT16, false); // 非连续
    aclTensor *other = CreateTensor(shape, &oAddr, ACL_FLOAT);
    aclTensor *out = CreateTensor(shape, &outAddr, ACL_FLOAT);
    float val = 1.0f;
    aclScalar *alpha = aclCreateScalar(&val, ACL_FLOAT);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &exec);
    if (wsSize > 0) aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    auto ret = aclnnAdd(wsAddr, wsSize, exec, stream);
    aclrtSynchronizeStream(stream);

    Check(ret == ACL_SUCCESS, "Add_MixedDtype_And_NonContiguous");

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out); aclDestroyScalar(alpha);
    aclrtFree(sAddr); aclrtFree(oAddr); aclrtFree(outAddr); if (wsSize > 0) aclrtFree(wsAddr);
}

void Test_Inplace_And_Errors() {
    // 覆盖: InplaceAdd, 以及 add.cpp 中的 OP_LOGE 异常校验分支
    std::vector<int64_t> s_shape = {2}, o_shape = {2, 2};
    void *sAddr = nullptr, *oAddr = nullptr;
    
    // 1. 制造 Broadcast Shape 不匹配的 Inplace 错误
    aclTensor *self1 = CreateTensor(s_shape, &sAddr, ACL_FLOAT);
    aclTensor *other1 = CreateTensor(o_shape, &oAddr, ACL_FLOAT);
    float val = 1.0f;
    aclScalar *alpha = aclCreateScalar(&val, ACL_FLOAT);
    
    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    auto ret1 = aclnnInplaceAddGetWorkspaceSize(self1, other1, alpha, &wsSize, &exec);
    Check(ret1 != ACL_SUCCESS, "InplaceAdd_Error_Broadcast_Mismatch"); // 预期失败
    
    aclDestroyTensor(self1); aclDestroyTensor(other1);
    aclrtFree(sAddr); aclrtFree(oAddr);

    // 2. 制造混合精度 Inplace 错误 (Out类型不能小于Other类型)
    aclTensor *self2 = CreateTensor(s_shape, &sAddr, ACL_FLOAT);
    aclTensor *other2 = CreateTensor(s_shape, &oAddr, ACL_FLOAT16);
    auto ret2 = aclnnInplaceAddGetWorkspaceSize(self2, other2, alpha, &wsSize, &exec);
    Check(ret2 != ACL_SUCCESS, "InplaceAdd_Error_MixDtype_Invalid"); // 预期失败
    
    aclDestroyTensor(self2); aclDestroyTensor(other2); aclDestroyScalar(alpha);
    aclrtFree(sAddr); aclrtFree(oAddr);
}

void Test_V3_Family(aclrtStream stream) {
    void *p = nullptr, *po = nullptr, *ws = nullptr;
    std::vector<int64_t> shape = {2, 2};
    float s_val = 10.0f, a_val = 1.0f;

    aclScalar *selfScalar = aclCreateScalar(&s_val, ACL_FLOAT);
    aclTensor *otherTensor = CreateTensor(shape, &p, ACL_FLOAT);
    aclScalar *alpha = aclCreateScalar(&a_val, ACL_FLOAT);
    aclTensor *out = CreateTensor(shape, &po, ACL_FLOAT);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    
    // AddV3 & InplaceAddV3
    auto r1 = aclnnAddV3GetWorkspaceSize(selfScalar, otherTensor, alpha, out, &wsSize, &exec);
    Check(r1 == 0, "AddV3_API");
    auto r2 = aclnnInplaceAddV3GetWorkspaceSize(selfScalar, otherTensor, alpha, &wsSize, &exec);
    Check(r2 == 0, "InplaceAddV3_API");

    // Adds (Tensor + Scalar)
    auto r3 = aclnnAddsGetWorkspaceSize(otherTensor, selfScalar, alpha, out, &wsSize, &exec);
    Check(r3 == 0, "Adds_API");

    aclDestroyScalar(selfScalar); aclDestroyTensor(otherTensor);
    aclDestroyScalar(alpha); aclDestroyTensor(out);
    aclrtFree(p); aclrtFree(po);
}

void Test_Other_APIs(aclrtStream stream) {
    // 覆盖: Adds, InplaceAdds, AddV3, InplaceAddV3
    void *tAddr = nullptr, *outAddr = nullptr, *wsAddr = nullptr;
    std::vector<int64_t> shape = {2};
    float val1 = 2.0f, val2 = 1.0f;
    
    aclTensor *tensor = CreateTensor(shape, &tAddr, ACL_FLOAT);
    aclTensor *out = CreateTensor(shape, &outAddr, ACL_FLOAT);
    aclScalar *otherScalar = aclCreateScalar(&val1, ACL_FLOAT);
    aclScalar *alpha = aclCreateScalar(&val2, ACL_FLOAT);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    
    // aclnnAdds
    aclnnAddsGetWorkspaceSize(tensor, otherScalar, alpha, out, &wsSize, &exec);
    Check(exec != nullptr, "Adds_API");

    // aclnnInplaceAdds
    aclnnInplaceAddsGetWorkspaceSize(tensor, otherScalar, alpha, &wsSize, &exec);
    Check(exec != nullptr, "InplaceAdds_API");

    // aclnnAddV3
    aclnnAddV3GetWorkspaceSize(otherScalar, tensor, alpha, out, &wsSize, &exec);
    Check(exec != nullptr, "AddV3_API");

    // aclnnInplaceAddV3
    aclnnInplaceAddV3GetWorkspaceSize(otherScalar, tensor, alpha, &wsSize, &exec);
    Check(exec != nullptr, "InplaceAddV3_API");

    aclDestroyTensor(tensor); aclDestroyTensor(out); 
    aclDestroyScalar(otherScalar); aclDestroyScalar(alpha);
    aclrtFree(tAddr); aclrtFree(outAddr);
}

int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    Test_Add_Standard_And_Axpy_Branches(stream);
    Test_MixDtype_And_NonContiguous(stream);
    Test_Inplace_And_Errors();
    Test_V3_Family(stream);
    Test_Other_APIs(stream);

    if (failCount == 0) {
        printf("Total Test Status: [PASS]\n");
    } else {
        printf("Total Test Status: [FAIL] (%d failed)\n", failCount);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return failCount == 0 ? 0 : -1;
}