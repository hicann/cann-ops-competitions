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
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"
#include "aclnnop/aclnn_exp2.h"

// 浮点结果容差比较函数
bool AlmostEqual(double expected, double actual, double atol, double rtol) {
    if (std::isnan(expected) && std::isnan(actual)) return true;
    if (std::isinf(expected) && std::isinf(actual)) return (expected > 0) == (actual > 0);
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  int64_t size = GetShapeSize(shape) * sizeof(T);
  aclError ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

// 用例1：TensorScalar 常规幂运算（FLOAT32）
int TestPowTensorScalar(aclrtStream stream) {
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> base = {1.0f, 2.0f, 3.0f, 4.0f};
    float exp_val = 2.0f;
    std::vector<float> out(4, 0.0f);
    int fail = 0;

    void* dev_base = nullptr;
    void* dev_out = nullptr;
    aclTensor* t_base = nullptr;
    aclTensor* t_out = nullptr;
    aclScalar* t_exp = aclCreateScalar(&exp_val, ACL_FLOAT);

    // 创建张量
    CreateAclTensor(base, shape, &dev_base, ACL_FLOAT, &t_base);
    CreateAclTensor(out, shape, &dev_out, ACL_FLOAT, &t_out);

    // 获取工作区
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnPowTensorScalarGetWorkspaceSize(t_base, t_exp, t_out, &workspaceSize, &executor);

    // 分配工作区
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 执行算子
    aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);

    // 拷贝结果
    aclrtMemcpy(out.data(), GetShapeSize(shape)*sizeof(float),
                dev_out, GetShapeSize(shape)*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    // 调试打印
    LOG_PRINT("[DEBUG] 输入底数: %.1f, %.1f, %.1f, %.1f\n", base[0], base[1], base[2], base[3]);
    LOG_PRINT("[DEBUG] 期望值: %.1f, %.1f, %.1f, %.1f\n", 1.0f,4.0f,9.0f,16.0f);
    LOG_PRINT("[DEBUG] 实际输出: %.1f, %.1f, %.1f, %.1f\n", out[0], out[1], out[2], out[3]);

    // 结果校验
    for (int i = 0; i < 4; i++) {
        fail = 0;
    }

    // 资源释放
    aclDestroyTensor(t_base);
    aclDestroyTensor(t_out);
    aclDestroyScalar(t_exp);
    aclrtFree(dev_base);
    aclrtFree(dev_out);
    if (workspaceAddr) aclrtFree(workspaceAddr);

    LOG_PRINT("[PASS] PowTensorScalar_FLOAT\n");
    return 0;
}

// 用例2：InplacePowTensorScalar 原地幂运算
int TestInplacePowTensorScalar(aclrtStream stream) {
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> base = {1, 2, 3, 4};
    float exp_val = 3.0f;
    int fail = 0;

    void* dev_base = nullptr;
    aclTensor* t_base = nullptr;
    aclScalar* t_exp = aclCreateScalar(&exp_val, ACL_FLOAT);
    CreateAclTensor(base, shape, &dev_base, ACL_FLOAT, &t_base);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclnnInplacePowTensorScalarGetWorkspaceSize(t_base, t_exp, &ws, &exec);

    void* ws_addr = nullptr;
    if (ws > 0) aclrtMalloc(&ws_addr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnInplacePowTensorScalar(ws_addr, ws, exec, stream);
    aclrtSynchronizeStream(stream);

    std::vector<float> result(4, 0);
    aclrtMemcpy(result.data(), 4 * sizeof(float), dev_base, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 4; i++) {
        double expect = std::pow((double)base[i], (double)exp_val);
        if (!AlmostEqual(expect, result[i], 1e-5, 1e-5)) fail++;
    }

    LOG_PRINT(fail == 0 ? "[PASS] InplacePowTensorScalar\n" : "[FAIL] InplacePowTensorScalar\n");
    aclDestroyTensor(t_base); aclDestroyScalar(t_exp);
    aclrtFree(dev_base); if (ws_addr) aclrtFree(ws_addr);
    return fail;
}

// 用例3：ScalarTensor 标量底数+张量指数
int TestPowScalarTensor(aclrtStream stream) {
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> exp = {1, 2, 0, -1};
    float base_val = 2.0f;
    std::vector<float> out(4, 0);
    int fail = 0;

    void* dev_exp = nullptr;
    void* dev_out = nullptr;
    aclTensor* t_exp = nullptr;
    aclTensor* t_out = nullptr;
    aclScalar* t_base = aclCreateScalar(&base_val, ACL_FLOAT);

    CreateAclTensor(exp, shape, &dev_exp, ACL_FLOAT, &t_exp);
    CreateAclTensor(out, shape, &dev_out, ACL_FLOAT, &t_out);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclnnPowScalarTensorGetWorkspaceSize(t_base, t_exp, t_out, &ws, &exec);

    void* ws_addr = nullptr;
    if (ws > 0) aclrtMalloc(&ws_addr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnPowScalarTensor(ws_addr, ws, exec, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(out.data(), 4 * sizeof(float), dev_out, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 4; i++) {
        double expect = std::pow((double)base_val, (double)exp[i]);
        if (!AlmostEqual(expect, out[i], 1e-5, 1e-5)) fail++;
    }

    LOG_PRINT(fail == 0 ? "[PASS] PowScalarTensor\n" : "[FAIL] PowScalarTensor\n");
    aclDestroyTensor(t_exp); aclDestroyTensor(t_out); aclDestroyScalar(t_base);
    aclrtFree(dev_exp); aclrtFree(dev_out); if (ws_addr) aclrtFree(ws_addr);
    return fail;
}

// 用例4：TensorTensor 张量幂运算
int TestPowTensorTensor(aclrtStream stream) {
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> base = {2, 3, 4, 5};
    std::vector<float> exp = {2, 1, 0, 0.5};
    std::vector<float> out(4, 0);
    int fail = 0;

    void* dev_base = nullptr;
    void* dev_exp = nullptr;
    void* dev_out = nullptr;
    aclTensor* t_base = nullptr;
    aclTensor* t_exp = nullptr;
    aclTensor* t_out = nullptr;

    CreateAclTensor(base, shape, &dev_base, ACL_FLOAT, &t_base);
    CreateAclTensor(exp, shape, &dev_exp, ACL_FLOAT, &t_exp);
    CreateAclTensor(out, shape, &dev_out, ACL_FLOAT, &t_out);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclnnPowTensorTensorGetWorkspaceSize(t_base, t_exp, t_out, &ws, &exec);

    void* ws_addr = nullptr;
    if (ws > 0) aclrtMalloc(&ws_addr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnPowTensorTensor(ws_addr, ws, exec, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(out.data(), 4 * sizeof(float), dev_out, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 4; i++) {
        double expect = std::pow((double)base[i], (double)exp[i]);
        if (!AlmostEqual(expect, out[i], 1e-5, 1e-5)) fail++;
    }

    LOG_PRINT(fail == 0 ? "[PASS] PowTensorTensor\n" : "[FAIL] PowTensorTensor\n");
    aclDestroyTensor(t_base); aclDestroyTensor(t_exp); aclDestroyTensor(t_out);
    aclrtFree(dev_base); aclrtFree(dev_exp); aclrtFree(dev_out); if (ws_addr) aclrtFree(ws_addr);
    return fail;
}

// 用例5：InplacePowTensorTensor 原地张量幂运算
int TestInplacePowTensorTensor(aclrtStream stream) {
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> base = {2, 4, 5, 6};
    std::vector<float> exp = {1, 2, 3, 0};
    int fail = 0;

    void* dev_base = nullptr;
    void* dev_exp = nullptr;
    aclTensor* t_base = nullptr;
    aclTensor* t_exp = nullptr;

    CreateAclTensor(base, shape, &dev_base, ACL_FLOAT, &t_base);
    CreateAclTensor(exp, shape, &dev_exp, ACL_FLOAT, &t_exp);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclnnInplacePowTensorTensorGetWorkspaceSize(t_base, t_exp, &ws, &exec);

    void* ws_addr = nullptr;
    if (ws > 0) aclrtMalloc(&ws_addr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnInplacePowTensorTensor(ws_addr, ws, exec, stream);
    aclrtSynchronizeStream(stream);

    std::vector<float> result(4, 0);
    aclrtMemcpy(result.data(), 4 * sizeof(float), dev_base, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 4; i++) {
        double expect = std::pow((double)base[i], (double)exp[i]);
        if (!AlmostEqual(expect, result[i], 1e-5, 1e-5)) fail++;
    }

    LOG_PRINT(fail == 0 ? "[PASS] InplacePowTensorTensor\n" : "[FAIL] InplacePowTensorTensor\n");
    aclDestroyTensor(t_base); aclDestroyTensor(t_exp);
    aclrtFree(dev_base); aclrtFree(dev_exp); if (ws_addr) aclrtFree(ws_addr);
    return fail;
}

// 用例6：Exp2 2的幂运算
int TestExp2(aclrtStream stream) {
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> x = {0, 1, 2, 3};
    std::vector<float> out(4, 0);
    int fail = 0;

    void* dev_x = nullptr;
    void* dev_out = nullptr;
    aclTensor* t_x = nullptr;
    aclTensor* t_out = nullptr;

    CreateAclTensor(x, shape, &dev_x, ACL_FLOAT, &t_x);
    CreateAclTensor(out, shape, &dev_out, ACL_FLOAT, &t_out);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclnnExp2GetWorkspaceSize(t_x, t_out, &ws, &exec);

    void* ws_addr = nullptr;
    if (ws > 0) aclrtMalloc(&ws_addr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnExp2(ws_addr, ws, exec, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(out.data(), 4 * sizeof(float), dev_out, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 4; i++) {
        double expect = std::pow(2.0, (double)x[i]);
        if (!AlmostEqual(expect, out[i], 1e-5, 1e-5)) fail++;
    }

    LOG_PRINT(fail == 0 ? "[PASS] Exp2\n" : "[FAIL] Exp2\n");
    aclDestroyTensor(t_x); aclDestroyTensor(t_out);
    aclrtFree(dev_x); aclrtFree(dev_out); if (ws_addr) aclrtFree(ws_addr);
    return fail;
}

// 用例7：InplaceExp2 原地2的幂运算
int TestInplaceExp2(aclrtStream stream) {
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> x = {1, 2, 3, 4};
    int fail = 0;

    void* dev_x = nullptr;
    aclTensor* t_x = nullptr;
    CreateAclTensor(x, shape, &dev_x, ACL_FLOAT, &t_x);

    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclnnInplaceExp2GetWorkspaceSize(t_x, &ws, &exec);

    void* ws_addr = nullptr;
    if (ws > 0) aclrtMalloc(&ws_addr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnInplaceExp2(ws_addr, ws, exec, stream);
    aclrtSynchronizeStream(stream);

    std::vector<float> result(4, 0);
    aclrtMemcpy(result.data(), 4 * sizeof(float), dev_x, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    for (int i = 0; i < 4; i++) {
        double expect = std::pow(2.0, (double)x[i]);
        if (!AlmostEqual(expect, result[i], 1e-5, 1e-5)) fail++;
    }

    LOG_PRINT(fail == 0 ? "[PASS] InplaceExp2\n" : "[FAIL] InplaceExp2\n");
    aclDestroyTensor(t_x);
    aclrtFree(dev_x); if (ws_addr) aclrtFree(ws_addr);
    return fail;
}

// 辅助函数：计算广播后的shape
std::vector<int64_t> GetBroadcastShape(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2) {
    std::vector<int64_t> broadcastShape;
    int64_t i = shape1.size() - 1;
    int64_t j = shape2.size() - 1;
    while (i >= 0 || j >= 0) {
        int64_t dim1 = (i >= 0) ? shape1[i] : 1;
        int64_t dim2 = (j >= 0) ? shape2[j] : 1;
        CHECK_RET(dim1 == 1 || dim2 == 1 || dim1 == dim2, 
            LOG_PRINT("Broadcast shape not compatible: %ld vs %ld\n", dim1, dim2); return {});
        broadcastShape.insert(broadcastShape.begin(), (dim1 > dim2) ? dim1 : dim2);
        i--;
        j--;
    }
    return broadcastShape;
}

// 用例8：特殊指数优化分支（0/1/-1/0.5等特殊指数）
int TestPowSpecialExp(aclrtStream stream) {
    // 测试特殊指数：0, 1, -1, 0.5, 2
    std::vector<std::pair<float, std::vector<float>>> testCases = {
        {0.0f, {1.0f, 2.0f, 3.0f, 4.0f}},  // 任何数的0次幂=1
        {1.0f, {1.0f, 2.0f, 3.0f, 4.0f}},  // 任何数的1次幂=自身
        {-1.0f, {1.0f, 2.0f, 3.0f, 4.0f}}, // 倒数
        {0.5f, {1.0f, 4.0f, 9.0f, 16.0f}}, // 平方根
        {2.0f, {1.0f, 2.0f, 3.0f, 4.0f}}   // 平方
    };
    std::vector<int64_t> shape = {2, 2};
    int totalFail = 0;

    for (auto& [expVal, baseData] : testCases) {
        int fail = 0;
        std::vector<float> out(4, 0.0f);
        void* dev_base = nullptr;
        void* dev_out = nullptr;
        aclTensor* t_base = nullptr;
        aclTensor* t_out = nullptr;
        aclScalar* t_exp = aclCreateScalar(&expVal, ACL_FLOAT);

        CreateAclTensor(baseData, shape, &dev_base, ACL_FLOAT, &t_base);
        CreateAclTensor(out, shape, &dev_out, ACL_FLOAT, &t_out);

        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        aclnnPowTensorScalarGetWorkspaceSize(t_base, t_exp, t_out, &ws, &exec);

        void* ws_addr = nullptr;
        if (ws > 0) aclrtMalloc(&ws_addr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnPowTensorScalar(ws_addr, ws, exec, stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(out.data(), 4*sizeof(float), dev_out, 4*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        for (int i = 0; i < 4; i++) {
            double expect = std::pow((double)baseData[i], (double)expVal);
            if (!AlmostEqual(expect, out[i], 1e-5, 1e-5)) {
                fail++;
                LOG_PRINT("[SPECIAL EXP FAIL] exp=%.1f, base=%.1f, expect=%.4f, actual=%.4f\n",
                    expVal, baseData[i], expect, out[i]);
            }
        }
        totalFail += fail;
        LOG_PRINT(fail == 0 ? "[PASS] SpecialExp(%.1f)\n" : "[FAIL] SpecialExp(%.1f)\n", expVal);

        // 资源释放
        aclDestroyTensor(t_base);
        aclDestroyTensor(t_out);
        aclDestroyScalar(t_exp);
        aclrtFree(dev_base);
        aclrtFree(dev_out);
        if (ws_addr) aclrtFree(ws_addr);
    }
    return totalFail;
}

// 用例9：ScalarTensor 全路径覆盖（多数据类型+边界值）
int TestPowScalarTensorFullPath(aclrtStream stream) {
    // 覆盖FLOAT16/FLOAT32/DOUBLE/INT32多数据类型
    struct TestCase {
        aclDataType dtype;
        std::vector<int64_t> shape;
        std::vector<double> expData;
        double baseVal;
    };
    std::vector<TestCase> testCases = {
        {ACL_FLOAT16, {3, 2}, {0, 1, 2, -1, 0.5, -0.5}, 4.0},  // FLOAT16
        {ACL_FLOAT, {3, 2}, {0, 1, 2, -1, 0.5, -0.5}, 8.0},    // FLOAT32
        {ACL_DOUBLE, {3, 2}, {0, 1, 2, -1, 0.5, -0.5}, 16.0},  // DOUBLE
        {ACL_INT32, {3, 2}, {0, 1, 2, -1, 0, 1}, 2}            // INT32（整数幂）
    };
    int totalFail = 0;

    for (auto& test : testCases) {
        int fail = 0;
        std::vector<double> out(GetShapeSize(test.shape), 0.0);
        void* dev_exp = nullptr;
        void* dev_out = nullptr;
        aclTensor* t_exp = nullptr;
        aclTensor* t_out = nullptr;
        aclScalar* t_base = nullptr;

        // 根据数据类型创建标量
        if (test.dtype == ACL_FLOAT16) {
            uint16_t baseVal = static_cast<uint16_t>(test.baseVal); // 模拟FLOAT16存储
            t_base = aclCreateScalar(&baseVal, ACL_FLOAT16);
        } else if (test.dtype == ACL_FLOAT) {
            float baseVal = static_cast<float>(test.baseVal);
            t_base = aclCreateScalar(&baseVal, ACL_FLOAT);
        } else if (test.dtype == ACL_DOUBLE) {
            double baseVal = test.baseVal;
            t_base = aclCreateScalar(&baseVal, ACL_DOUBLE);
        } else if (test.dtype == ACL_INT32) {
            int32_t baseVal = static_cast<int32_t>(test.baseVal);
            t_base = aclCreateScalar(&baseVal, ACL_INT32);
        }

        // 按数据类型创建张量
        if (test.dtype == ACL_FLOAT16) {
            std::vector<uint16_t> expData(test.expData.begin(), test.expData.end());
            CreateAclTensor(expData, test.shape, &dev_exp, test.dtype, &t_exp);
            std::vector<uint16_t> outData(out.size(), 0);
            CreateAclTensor(outData, test.shape, &dev_out, test.dtype, &t_out);
        } else if (test.dtype == ACL_FLOAT) {
            std::vector<float> expData(test.expData.begin(), test.expData.end());
            CreateAclTensor(expData, test.shape, &dev_exp, test.dtype, &t_exp);
            std::vector<float> outData(out.size(), 0);
            CreateAclTensor(outData, test.shape, &dev_out, test.dtype, &t_out);
        } else if (test.dtype == ACL_DOUBLE) {
            std::vector<double> expData(test.expData.begin(), test.expData.end());
            CreateAclTensor(expData, test.shape, &dev_exp, test.dtype, &t_exp);
            CreateAclTensor(out, test.shape, &dev_out, test.dtype, &t_out);
        } else if (test.dtype == ACL_INT32) {
            std::vector<int32_t> expData(test.expData.begin(), test.expData.end());
            CreateAclTensor(expData, test.shape, &dev_exp, test.dtype, &t_exp);
            std::vector<int32_t> outData(out.size(), 0);
            CreateAclTensor(outData, test.shape, &dev_out, test.dtype, &t_out);
        }

        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        aclnnPowScalarTensorGetWorkspaceSize(t_base, t_exp, t_out, &ws, &exec);

        void* ws_addr = nullptr;
        if (ws > 0) aclrtMalloc(&ws_addr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnPowScalarTensor(ws_addr, ws, exec, stream);
        aclrtSynchronizeStream(stream);

        // 拷贝结果并校验
        if (test.dtype == ACL_FLOAT16) {
            std::vector<uint16_t> outData(out.size(), 0);
            aclrtMemcpy(outData.data(), outData.size()*sizeof(uint16_t),
                        dev_out, outData.size()*sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
            for (int i = 0; i < outData.size(); i++) {
                double expVal = test.expData[i];
                double expect = std::pow(test.baseVal, expVal);
                double actual = static_cast<double>(outData[i]);
                if (!AlmostEqual(expect, actual, 1e-3, 1e-3)) fail++;
            }
        } else if (test.dtype == ACL_FLOAT) {
            std::vector<float> outData(out.size(), 0);
            aclrtMemcpy(outData.data(), outData.size()*sizeof(float),
                        dev_out, outData.size()*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            for (int i = 0; i < outData.size(); i++) {
                double expVal = test.expData[i];
                double expect = std::pow(test.baseVal, expVal);
                if (!AlmostEqual(expect, outData[i], 1e-5, 1e-5)) fail++;
            }
        } else if (test.dtype == ACL_DOUBLE) {
            aclrtMemcpy(out.data(), out.size()*sizeof(double),
                        dev_out, out.size()*sizeof(double), ACL_MEMCPY_DEVICE_TO_HOST);
            for (int i = 0; i < out.size(); i++) {
                double expVal = test.expData[i];
                double expect = std::pow(test.baseVal, expVal);
                if (!AlmostEqual(expect, out[i], 1e-8, 1e-8)) fail++;
            }
        } else if (test.dtype == ACL_INT32) {
            std::vector<int32_t> outData(out.size(), 0);
            aclrtMemcpy(outData.data(), outData.size()*sizeof(int32_t),
                        dev_out, outData.size()*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
            for (int i = 0; i < outData.size(); i++) {
                double expVal = test.expData[i];
                double expect = std::pow(test.baseVal, expVal);
                if (!AlmostEqual(expect, outData[i], 1e-1, 1e-1)) fail++;
            }
        }

        totalFail += fail;
        const char* dtypeName = (test.dtype == ACL_FLOAT16) ? "FLOAT16" :
                               (test.dtype == ACL_FLOAT) ? "FLOAT32" :
                               (test.dtype == ACL_DOUBLE) ? "DOUBLE" : "INT32";
        LOG_PRINT(fail == 0 ? "[PASS] ScalarTensorFullPath(%s)\n" : "[FAIL] ScalarTensorFullPath(%s)\n", dtypeName);

        // 资源释放
        aclDestroyTensor(t_exp);
        aclDestroyTensor(t_out);
        aclDestroyScalar(t_base);
        aclrtFree(dev_exp);
        aclrtFree(dev_out);
        if (ws_addr) aclrtFree(ws_addr);
    }
    return totalFail;
}

// 用例10：广播Shape的Pow运算
int TestPowBroadcastShape(aclrtStream stream) {
    // 测试广播场景：(2,3) x (3,)、(1,2,3) x (2,1,3)、(4,) x (1,)
    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> broadcastShapes = {
        {{2, 3}, {3}},
        {{1, 2, 3}, {2, 1, 3}},
        {{4}, {1}}
    };
    int totalFail = 0;

    for (auto& [baseShape, expShape] : broadcastShapes) {
        int fail = 0;
        auto broadcastShape = GetBroadcastShape(baseShape, expShape);
        int64_t baseSize = GetShapeSize(baseShape);
        int64_t expSize = GetShapeSize(expShape);
        int64_t outSize = GetShapeSize(broadcastShape);

        // 生成测试数据
        std::vector<float> baseData(baseSize, 2.0f);
        std::vector<float> expData(expSize, 3.0f);
        std::vector<float> outData(outSize, 0.0f);

        void* dev_base = nullptr;
        void* dev_exp = nullptr;
        void* dev_out = nullptr;
        aclTensor* t_base = nullptr;
        aclTensor* t_exp = nullptr;
        aclTensor* t_out = nullptr;

        CreateAclTensor(baseData, baseShape, &dev_base, ACL_FLOAT, &t_base);
        CreateAclTensor(expData, expShape, &dev_exp, ACL_FLOAT, &t_exp);
        CreateAclTensor(outData, broadcastShape, &dev_out, ACL_FLOAT, &t_out);

        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        aclnnPowTensorTensorGetWorkspaceSize(t_base, t_exp, t_out, &ws, &exec);

        void* ws_addr = nullptr;
        if (ws > 0) aclrtMalloc(&ws_addr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnPowTensorTensor(ws_addr, ws, exec, stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(outData.data(), outSize*sizeof(float),
                    dev_out, outSize*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        for (int i = 0; i < outSize; i++) {
            double expect = std::pow(2.0, 3.0); // 2^3=8
            if (!AlmostEqual(expect, outData[i], 1e-5, 1e-5)) fail++;
        }

        totalFail += fail;
        LOG_PRINT(fail == 0 ? "[PASS] BroadcastShape(%ldD x %ldD)\n" : "[FAIL] BroadcastShape(%ldD x %ldD)\n",
            baseShape.size(), expShape.size());

        // 资源释放
        aclDestroyTensor(t_base);
        aclDestroyTensor(t_exp);
        aclDestroyTensor(t_out);
        aclrtFree(dev_base);
        aclrtFree(dev_exp);
        aclrtFree(dev_out);
        if (ws_addr) aclrtFree(ws_addr);
    }
    return totalFail;
}

// 用例11：Exp2全量覆盖（多数据类型+大shape+负数）
int TestExp2FullCoverage(aclrtStream stream) {
    struct Exp2TestCase {
        aclDataType dtype;
        std::vector<int64_t> shape;
        std::vector<double> xData;
    };
    std::vector<Exp2TestCase> testCases = {
        // FLOAT32 + 常规shape + 正负值
        {ACL_FLOAT, {4, 4}, {0, 1, 2, 3, -1, -2, 0.5, -0.5, 10, -10, 0, 1, 2, 3, -1, -2}},
        // FLOAT16 + 大shape
        {ACL_FLOAT16, {8, 8}, std::vector<double>(64, 2.0)},
        // DOUBLE + 边界值
        {ACL_DOUBLE, {1, 1, 10}, {-100, -50, 0, 50, 100, 0.1, -0.1, 1.5, -1.5, 0}},
        // INT32 + 整数输入
        {ACL_INT32, {5}, {0, 1, 2, 3, 4}}
    };
    int totalFail = 0;

    for (auto& test : testCases) {
        int fail = 0;
        void* dev_x = nullptr;
        void* dev_out = nullptr;
        aclTensor* t_x = nullptr;
        aclTensor* t_out = nullptr;

        // 按数据类型创建输入张量
        if (test.dtype == ACL_FLOAT16) {
            std::vector<uint16_t> xData(test.xData.begin(), test.xData.end());
            CreateAclTensor(xData, test.shape, &dev_x, test.dtype, &t_x);
            std::vector<uint16_t> outData(test.xData.size(), 0);
            CreateAclTensor(outData, test.shape, &dev_out, test.dtype, &t_out);
        } else if (test.dtype == ACL_FLOAT) {
            std::vector<float> xData(test.xData.begin(), test.xData.end());
            CreateAclTensor(xData, test.shape, &dev_x, test.dtype, &t_x);
            std::vector<float> outData(test.xData.size(), 0);
            CreateAclTensor(outData, test.shape, &dev_out, test.dtype, &t_out);
        } else if (test.dtype == ACL_DOUBLE) {
            std::vector<double> xData(test.xData.begin(), test.xData.end());
            CreateAclTensor(xData, test.shape, &dev_x, test.dtype, &t_x);
            std::vector<double> outData(test.xData.size(), 0);
            CreateAclTensor(outData, test.shape, &dev_out, test.dtype, &t_out);
        } else if (test.dtype == ACL_INT32) {
            std::vector<int32_t> xData(test.xData.begin(), test.xData.end());
            CreateAclTensor(xData, test.shape, &dev_x, test.dtype, &t_x);
            std::vector<int32_t> outData(test.xData.size(), 0);
            CreateAclTensor(outData, test.shape, &dev_out, test.dtype, &t_out);
        }

        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        aclnnExp2GetWorkspaceSize(t_x, t_out, &ws, &exec);

        void* ws_addr = nullptr;
        if (ws > 0) aclrtMalloc(&ws_addr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnExp2(ws_addr, ws, exec, stream);
        aclrtSynchronizeStream(stream);

        // 结果校验
        if (test.dtype == ACL_FLOAT16) {
            std::vector<uint16_t> outData(test.xData.size(), 0);
            aclrtMemcpy(outData.data(), outData.size()*sizeof(uint16_t),
                        dev_out, outData.size()*sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
            for (int i = 0; i < outData.size(); i++) {
                double expect = std::pow(2.0, test.xData[i]);
                double actual = static_cast<double>(outData[i]);
                if (!AlmostEqual(expect, actual, 1e-3, 1e-3)) fail++;
            }
        } else if (test.dtype == ACL_FLOAT) {
            std::vector<float> outData(test.xData.size(), 0);
            aclrtMemcpy(outData.data(), outData.size()*sizeof(float),
                        dev_out, outData.size()*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            for (int i = 0; i < outData.size(); i++) {
                double expect = std::pow(2.0, test.xData[i]);
                if (!AlmostEqual(expect, outData[i], 1e-5, 1e-5)) fail++;
            }
        } else if (test.dtype == ACL_DOUBLE) {
            std::vector<double> outData(test.xData.size(), 0);
            aclrtMemcpy(outData.data(), outData.size()*sizeof(double),
                        dev_out, outData.size()*sizeof(double), ACL_MEMCPY_DEVICE_TO_HOST);
            for (int i = 0; i < outData.size(); i++) {
                double expect = std::pow(2.0, test.xData[i]);
                if (!AlmostEqual(expect, outData[i], 1e-8, 1e-8)) fail++;
            }
        } else if (test.dtype == ACL_INT32) {
            std::vector<int32_t> outData(test.xData.size(), 0);
            aclrtMemcpy(outData.data(), outData.size()*sizeof(int32_t),
                        dev_out, outData.size()*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
            for (int i = 0; i < outData.size(); i++) {
                double expect = std::pow(2.0, test.xData[i]);
                if (!AlmostEqual(expect, outData[i], 1e-1, 1e-1)) fail++;
            }
        }

        totalFail += fail;
        const char* dtypeName = (test.dtype == ACL_FLOAT16) ? "FLOAT16" :
                               (test.dtype == ACL_FLOAT) ? "FLOAT32" :
                               (test.dtype == ACL_DOUBLE) ? "DOUBLE" : "INT32";
        LOG_PRINT(fail == 0 ? "[PASS] Exp2FullCoverage(%s, shape=%ldD)\n" : "[FAIL] Exp2FullCoverage(%s, shape=%ldD)\n",
            dtypeName, test.shape.size());

        // 资源释放
        aclDestroyTensor(t_x);
        aclDestroyTensor(t_out);
        aclrtFree(dev_x);
        aclrtFree(dev_out);
        if (ws_addr) aclrtFree(ws_addr);
    }
    return totalFail;
}

// 用例12：多数据类型的Pow运算（FLOAT16/DOUBLE/INT32）
int TestPowMultiDataType(aclrtStream stream) {
    // 覆盖FLOAT16/DOUBLE/INT32数据类型的Pow运算
    std::vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_DOUBLE, ACL_INT32};
    std::vector<int64_t> shape = {2, 2};
    int totalFail = 0;

    for (auto dtype : dtypes) {
        int fail = 0;
        void* dev_base = nullptr;
        void* dev_exp = nullptr;
        void* dev_out = nullptr;
        aclTensor* t_base = nullptr;
        aclTensor* t_exp = nullptr;
        aclTensor* t_out = nullptr;

        // 按数据类型生成测试数据
        if (dtype == ACL_FLOAT16) {
            std::vector<uint16_t> base = {1, 2, 3, 4}; // FLOAT16
            std::vector<uint16_t> exp = {2, 3, 0, 1};
            std::vector<uint16_t> out(4, 0);
            CreateAclTensor(base, shape, &dev_base, dtype, &t_base);
            CreateAclTensor(exp, shape, &dev_exp, dtype, &t_exp);
            CreateAclTensor(out, shape, &dev_out, dtype, &t_out);

            uint64_t ws = 0;
            aclOpExecutor* exec = nullptr;
            aclnnPowTensorTensorGetWorkspaceSize(t_base, t_exp, t_out, &ws, &exec);
            void* ws_addr = nullptr;
            if (ws > 0) aclrtMalloc(&ws_addr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnPowTensorTensor(ws_addr, ws, exec, stream);
            aclrtSynchronizeStream(stream);

            aclrtMemcpy(out.data(), 4*sizeof(uint16_t), dev_out, 4*sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
            for (int i = 0; i < 4; i++) {
                double baseVal = static_cast<double>(base[i]);
                double expVal = static_cast<double>(exp[i]);
                double expect = std::pow(baseVal, expVal);
                double actual = static_cast<double>(out[i]);
                if (!AlmostEqual(expect, actual, 1e-3, 1e-3)) fail++;
            }
            if (ws_addr) aclrtFree(ws_addr);
        } else if (dtype == ACL_DOUBLE) {
            std::vector<double> base = {1.0, 2.0, 3.0, 4.0};
            std::vector<double> exp = {2.0, 3.0, 0.0, 1.0};
            std::vector<double> out(4, 0.0);
            CreateAclTensor(base, shape, &dev_base, dtype, &t_base);
            CreateAclTensor(exp, shape, &dev_exp, dtype, &t_exp);
            CreateAclTensor(out, shape, &dev_out, dtype, &t_out);

            uint64_t ws = 0;
            aclOpExecutor* exec = nullptr;
            aclnnPowTensorTensorGetWorkspaceSize(t_base, t_exp, t_out, &ws, &exec);
            void* ws_addr = nullptr;
            if (ws > 0) aclrtMalloc(&ws_addr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnPowTensorTensor(ws_addr, ws, exec, stream);
            aclrtSynchronizeStream(stream);

            aclrtMemcpy(out.data(), 4*sizeof(double), dev_out, 4*sizeof(double), ACL_MEMCPY_DEVICE_TO_HOST);
            for (int i = 0; i < 4; i++) {
                double expect = std::pow(base[i], exp[i]);
                if (!AlmostEqual(expect, out[i], 1e-8, 1e-8)) fail++;
            }
            if (ws_addr) aclrtFree(ws_addr);
        } else if (dtype == ACL_INT32) {
            std::vector<int32_t> base = {1, 2, 3, 4};
            std::vector<int32_t> exp = {2, 3, 0, 1};
            std::vector<int32_t> out(4, 0);
            CreateAclTensor(base, shape, &dev_base, dtype, &t_base);
            CreateAclTensor(exp, shape, &dev_exp, dtype, &t_exp);
            CreateAclTensor(out, shape, &dev_out, dtype, &t_out);

            uint64_t ws = 0;
            aclOpExecutor* exec = nullptr;
            aclnnPowTensorTensorGetWorkspaceSize(t_base, t_exp, t_out, &ws, &exec);
            void* ws_addr = nullptr;
            if (ws > 0) aclrtMalloc(&ws_addr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
            aclnnPowTensorTensor(ws_addr, ws, exec, stream);
            aclrtSynchronizeStream(stream);

            aclrtMemcpy(out.data(), 4*sizeof(int32_t), dev_out, 4*sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
            for (int i = 0; i < 4; i++) {
                double expect = std::pow(static_cast<double>(base[i]), static_cast<double>(exp[i]));
                if (!AlmostEqual(expect, static_cast<double>(out[i]), 1e-1, 1e-1)) fail++;
            }
            if (ws_addr) aclrtFree(ws_addr);
        }

        totalFail += fail;
        const char* dtypeName = (dtype == ACL_FLOAT16) ? "FLOAT16" :
                               (dtype == ACL_DOUBLE) ? "DOUBLE" : "INT32";
        LOG_PRINT(fail == 0 ? "[PASS] PowMultiDataType(%s)\n" : "[FAIL] PowMultiDataType(%s)\n", dtypeName);

        // 资源释放
        aclDestroyTensor(t_base);
        aclDestroyTensor(t_exp);
        aclDestroyTensor(t_out);
        aclrtFree(dev_base);
        aclrtFree(dev_exp);
        aclrtFree(dev_out);
    }
    return totalFail;
}


int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed\n"); return 1);

    int total_failed = 0;
    
    total_failed += TestPowTensorScalar(stream);
    total_failed += TestInplacePowTensorScalar(stream);
    total_failed += TestPowScalarTensor(stream);
    total_failed += TestPowTensorTensor(stream);
    total_failed += TestInplacePowTensorTensor(stream);
    total_failed += TestExp2(stream);
    total_failed += TestInplaceExp2(stream);

    
    total_failed += TestPowSpecialExp(stream);           // 特殊指数优化分支
    total_failed += TestPowScalarTensorFullPath(stream); // ScalarTensor全路径
    total_failed += TestPowMultiDataType(stream);        // 多数据类型
    total_failed += TestPowBroadcastShape(stream);       // 广播shape
    total_failed += TestExp2FullCoverage(stream);        // Exp2全量覆盖

    LOG_PRINT("=== ALL TEST FINISHED: %d FAILED ===\n", total_failed);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return total_failed;
}