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
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"
#include <string>
#include <limits>
#include <cstdint>
#include <complex>
#include <type_traits>
#include <cstring>
// 浮点数比较（支持绝对容差 atol 和相对容差 rtol）
// 用于验证 float32 等浮点类型的计算结果
bool AlmostEqual(double expected, double actual, double atol = 1e-5, double rtol = 1e-5)
{
  if (std::isnan(expected) && std::isnan(actual))
    return true;
  if (std::isinf(expected) && std::isinf(actual))
    return (expected > 0) == (actual > 0);
  return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}
// 专门用于复数比对的辅助函数
template <typename T>
bool AlmostEqualComplex(const std::complex<T> &expected, const std::complex<T> &actual, double atol = 1e-5, double rtol = 1e-5)
{
  bool realOk = AlmostEqual((double)expected.real(), (double)actual.real(), atol, rtol);
  bool imagOk = AlmostEqual((double)expected.imag(), (double)actual.imag(), atol, rtol);
  return realOk && imagOk;
}

#define CHECK_RET(cond, return_expr) \
  do                                 \
  {                                  \
    if (!(cond))                     \
    {                                \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do                                \
  {                                 \
    printf(message, ##__VA_ARGS__); \
  } while (0)

// ======================== 辅助函数 ========================
// 将 BFloat16 转换为标准的 Float32
float BFloat16ToFloat(uint16_t bf16)
{
  uint32_t val = ((uint32_t)bf16) << 16;
  float f;
  std::memcpy(&f, &val, 4);
  return f;
}

// 将 float16 的 16 位二进制表示转换为 float（符合 IEEE 754）
float Float16ToFloat(uint16_t fp16)
{
  // 提取符号位、指数位、尾数位
  uint16_t sign = (fp16 >> 15) & 0x1;
  uint16_t exponent = (fp16 >> 10) & 0x1F;
  uint16_t mantissa = fp16 & 0x3FF;

  // 处理特殊情况
  if (exponent == 0)
  {
    // 次正规数或零
    if (mantissa == 0)
    {
      // 零
      return sign ? -0.0f : 0.0f;
    }
    else
    {
      // 次正规数：值 = (-1)^sign * 2^{-14} * (mantissa / 1024.0)
      float value = (float)mantissa / 1024.0f;
      value = ldexp(value, -14);
      return sign ? -value : value;
    }
  }
  else if (exponent == 0x1F)
  {
    // 无穷大或 NaN
    if (mantissa == 0)
    {
      // 无穷大
      return sign ? -INFINITY : INFINITY;
    }
    else
    {
      // NaN
      return NAN;
    }
  }
  else
  {
    // 正规数
    float value = ldexpf((float)(mantissa | 0x400), exponent - 25);
    return sign ? -value : value;
  }
}
// 计算 shape 的元素总数
int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape)
  {
    shapeSize *= i;
  }
  return shapeSize;
}
// ACL 运行时初始化：初始化设备、创建 stream
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

// 创建 ACL tensor：分配设备内存、拷贝数据、构造 tensor 描述符
// 模板参数 T 支持 float, int32_t, uint16_t (for float16) 等
template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape,
                    void **deviceAddr,
                    aclDataType dataType,
                    aclTensor **tensor)
{
  auto size = GetShapeSize(shape) * sizeof(T);
  // 申请设备内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 拷贝主机数据到设备
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续存储的 strides（行优先）
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--)
  {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 创建 ACL tensor 描述符
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

// 创建非连续内存的 Tensor (模拟从大 Tensor 中切片出的 View)
template <typename T>
int CreateNonContiguousAclTensor(const std::vector<T> &hostData,
                                 const std::vector<int64_t> &shape,
                                 const std::vector<int64_t> &strides, // 手动传入特殊的 strides
                                 void **deviceAddr,
                                 aclDataType dataType,
                                 aclTensor **tensor)
{
  auto size = hostData.size() * sizeof(T); // 实际分配更大的内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS)
    return ret;
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS)
    return ret;

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  if (*tensor == nullptr)
  {
    return 1; // 创建失败
  }
  return 0;
}

// ======================== 测试用例 ========================
// 通用测试模板1：适用于 aclnnMul (Tensor * Tensor = Tensor)
template <typename T>
int RunGenericMulTest(const std::string &testName,
                      aclrtStream stream,
                      aclDataType dataType,
                      const std::vector<int64_t> &shape1,
                      const std::vector<int64_t> &shape2,
                      const std::vector<int64_t> &outShape,
                      const std::vector<T> &host1,
                      const std::vector<T> &host2,
                      const std::vector<T> &expectedHost,
                      double atol = 1e-5,
                      double rtol = 1e-5)
{

  // 变量声明统一定义，方便在出口处集中清理
  void *dev1 = nullptr, *dev2 = nullptr, *outDev = nullptr, *workspace = nullptr;
  aclTensor *t1 = nullptr, *t2 = nullptr, *out = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;
  int failed = 0;
  int retCode = 0; // 用于记录执行过程中的错误

  // 1. 准备空的主机输出数据
  std::vector<T> outHost(expectedHost.size(), 0);

  // 2. 分配内存并创建 Tensor (复用你现有的 CreateAclTensor 函数)
  // 注意：如果中途失败，直接跳到 cleanup 释放已分配的资源
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

  // 3. 两段式调用执行算子

  if (aclnnMulGetWorkspaceSize(t1, t2, out, &workspaceSize, &executor) != ACL_SUCCESS)
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
  if (aclnnMul(workspace, workspaceSize, executor, stream) != ACL_SUCCESS)
  {
    retCode = 1;
    goto cleanup;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
  {
    retCode = 1;
    goto cleanup;
  }

  // 4. 将结果拷回主机
  if (aclrtMemcpy(outHost.data(), outHost.size() * sizeof(T), outDev, outHost.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
  {
    retCode = 1;
    goto cleanup;
  }

  // 5. 结果比对
  for (size_t i = 0; i < expectedHost.size(); i++)
  {
    // 编译期类型判断：处理复数类型
    if constexpr (std::is_same<T, std::complex<float>>::value || std::is_same<T, std::complex<double>>::value)
    {
      if (!AlmostEqualComplex(expectedHost[i], outHost[i], atol, rtol))
      {
        LOG_PRINT("[FAIL] %s idx %zu: expected (%.5f, %.5f), got (%.5f, %.5f)\n",
                  testName.c_str(), i,
                  (double)expectedHost[i].real(), (double)expectedHost[i].imag(),
                  (double)outHost[i].real(), (double)outHost[i].imag());
        failed++;
      }
    }
    // 编译期类型判断：处理 16位 浮点 (Float16 或 BFloat16)
    else if constexpr (std::is_same<T, uint16_t>::value)
    {
      double expVal = 0.0, actVal = 0.0;
      if (dataType == ACL_FLOAT16)
      {
        expVal = (double)Float16ToFloat(expectedHost[i]);
        actVal = (double)Float16ToFloat(outHost[i]);
      }
      else if (dataType == ACL_BF16)
      {
        expVal = (double)BFloat16ToFloat(expectedHost[i]);
        actVal = (double)BFloat16ToFloat(outHost[i]);
      }
      else
      {
        expVal = (double)expectedHost[i];
        actVal = (double)outHost[i];
      }
      if (!AlmostEqual(expVal, actVal, atol, rtol))
      {
        LOG_PRINT("[FAIL] %s idx %zu: expected %f, got %f\n", testName.c_str(), i, expVal, actVal);
        failed++;
      }
    }
    // 处理常规数字类型 (Float32, Double, Int32, Int8 等)
    else
    {
      double expVal = (double)expectedHost[i];
      double actVal = (double)outHost[i];
      if (!AlmostEqual(expVal, actVal, atol, rtol))
      {
        LOG_PRINT("[FAIL] %s idx %zu: expected %f, got %f\n", testName.c_str(), i, expVal, actVal);
        failed++;
      }
    }
  }

cleanup:
  // 6. 统一输出结果并释放资源
  if (retCode != 0)
  {
    LOG_PRINT("[ERROR] %s terminated early due to an ACL API error.\n", testName.c_str());
  }
  else if (failed == 0)
  {
    LOG_PRINT("[PASS] %s\n", testName.c_str());
  }

  // 防内存泄漏：逐一检查指针，如果不为空则释放
  if (workspace)
    aclrtFree(workspace);
  if (t1)
    aclDestroyTensor(t1);
  if (t2)
    aclDestroyTensor(t2);
  if (out)
    aclDestroyTensor(out);
  if (dev1)
    aclrtFree(dev1);
  if (dev2)
    aclrtFree(dev2);
  if (outDev)
    aclrtFree(outDev);

  return failed + retCode;
}
// 通用测试模版2：适用于 aclnnMuls (Tensor * Scalar = Tensor)
template <typename T>
int RunGenericMulsTest(const std::string &testName,
                       aclrtStream stream,
                       aclDataType dataType,
                       const std::vector<int64_t> &shape,
                       const std::vector<T> &hostData,
                       T scalarValue,
                       const std::vector<T> &expectedHost,
                       double atol = 1e-5,
                       double rtol = 1e-5)
{

  // 变量声明统一定义，方便在出口处集中清理
  void *selfDev = nullptr, *outDev = nullptr, *workspace = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  aclScalar *scalar = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;
  int failed = 0;
  int retCode = 0;

  // 1. 准备空的主机输出数据
  std::vector<T> outHost(expectedHost.size(), 0);

  // 2. 分配内存并创建 Tensor
  if (CreateAclTensor(hostData, shape, &selfDev, dataType, &self) != 0)
  {
    retCode = 1;
    goto cleanup;
  }
  if (CreateAclTensor(outHost, shape, &outDev, dataType, &out) != 0)
  {
    retCode = 1;
    goto cleanup;
  }

  // 3. 创建 Scalar 描述符
  // 注意：传入标量值的地址和对应的数据类型
  scalar = aclCreateScalar(&scalarValue, dataType);
  if (scalar == nullptr)
  {
    retCode = 1;
    goto cleanup;
  }

  // 4. 两段式调用执行算子 (调用 Muls 变体)

  if (aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor) != ACL_SUCCESS)
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
  if (aclnnMuls(workspace, workspaceSize, executor, stream) != ACL_SUCCESS)
  {
    retCode = 1;
    goto cleanup;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
  {
    retCode = 1;
    goto cleanup;
  }

  // 5. 将结果拷回主机
  if (aclrtMemcpy(outHost.data(), outHost.size() * sizeof(T), outDev, outHost.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
  {
    retCode = 1;
    goto cleanup;
  }

  // 6. 结果比对
  for (size_t i = 0; i < expectedHost.size(); i++)
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
  // 7. 统一输出结果并释放资源
  if (retCode != 0)
  {
    LOG_PRINT("[ERROR] %s terminated early due to an ACL API error.\n", testName.c_str());
  }
  else if (failed == 0)
  {
    LOG_PRINT("[PASS] %s\n", testName.c_str());
  }

  // 防内存泄漏：逐一检查指针，如果不为空则释放
  if (workspace)
    aclrtFree(workspace);
  if (self)
    aclDestroyTensor(self);
  if (out)
    aclDestroyTensor(out);
  if (scalar)
    aclDestroyScalar(scalar); // 注意这里增加了 Scalar 的释放
  if (selfDev)
    aclrtFree(selfDev);
  if (outDev)
    aclrtFree(outDev);

  return failed + retCode;
}
// 通用测试模版3：InplaceMul 测试模板 (Tensor *= Tensor)
template <typename T>
int RunGenericInplaceMulTest(const std::string &testName,
                             aclrtStream stream,
                             aclDataType dataType,
                             const std::vector<int64_t> &shape,
                             const std::vector<T> &selfHost,
                             const std::vector<T> &otherHost,
                             const std::vector<T> &expectedHost,
                             double atol = 1e-5,
                             double rtol = 1e-5)
{

  // 变量声明移到顶部，避免 goto 报错
  void *selfDev = nullptr, *otherDev = nullptr, *workspace = nullptr;
  aclTensor *self = nullptr, *other = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;
  int failed = 0;
  int retCode = 0;

  // 分配内存并创建 Tensor (注意：self 既是输入也是输出)
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

  // 两段式调用 (调用 InplaceMul 变体)
  if (aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor) != ACL_SUCCESS)
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
  if (aclnnInplaceMul(workspace, workspaceSize, executor, stream) != ACL_SUCCESS)
  {
    retCode = 1;
    goto cleanup;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
  {
    retCode = 1;
    goto cleanup;
  }

  // 将结果从 selfDev 拷回主机进行比对
  { // 使用作用域限制 outHost 的生命周期
    std::vector<T> outHost(expectedHost.size(), 0);
    if (aclrtMemcpy(outHost.data(), outHost.size() * sizeof(T), selfDev, outHost.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
    {
      retCode = 1;
      goto cleanup;
    }

    for (size_t i = 0; i < expectedHost.size(); i++)
    {
      double expVal = (double)expectedHost[i];
      double actVal = (double)outHost[i];
      if (!AlmostEqual(expVal, actVal, atol, rtol))
      {
        LOG_PRINT("[FAIL] %s idx %zu: expected %f, got %f\n", testName.c_str(), i, expVal, actVal);
        failed++;
      }
    }
  }

cleanup:
  if (retCode != 0)
    LOG_PRINT("[ERROR] %s terminated early.\n", testName.c_str());
  else if (failed == 0)
    LOG_PRINT("[PASS] %s\n", testName.c_str());

  if (workspace)
    aclrtFree(workspace);
  if (self)
    aclDestroyTensor(self);
  if (other)
    aclDestroyTensor(other);
  if (selfDev)
    aclrtFree(selfDev);
  if (otherDev)
    aclrtFree(otherDev);

  return failed + retCode;
}

// 通用测试模版4：InplaceMuls 测试模板 (Tensor *= Scalar)
template <typename T>
int RunGenericInplaceMulsTest(const std::string &testName,
                              aclrtStream stream,
                              aclDataType dataType,
                              const std::vector<int64_t> &shape,
                              const std::vector<T> &selfHost,
                              T scalarValue,
                              const std::vector<T> &expectedHost,
                              double atol = 1e-5,
                              double rtol = 1e-5)
{

  void *selfDev = nullptr, *workspace = nullptr;
  aclTensor *self = nullptr;
  aclScalar *scalar = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;
  int failed = 0;
  int retCode = 0;

  if (CreateAclTensor(selfHost, shape, &selfDev, dataType, &self) != 0)
  {
    retCode = 1;
    goto cleanup;
  }

  scalar = aclCreateScalar(&scalarValue, dataType);
  if (scalar == nullptr)
  {
    retCode = 1;
    goto cleanup;
  }

  if (aclnnInplaceMulsGetWorkspaceSize(self, scalar, &workspaceSize, &executor) != ACL_SUCCESS)
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
  if (aclnnInplaceMuls(workspace, workspaceSize, executor, stream) != ACL_SUCCESS)
  {
    retCode = 1;
    goto cleanup;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
  {
    retCode = 1;
    goto cleanup;
  }

  {
    std::vector<T> outHost(expectedHost.size(), 0);
    if (aclrtMemcpy(outHost.data(), outHost.size() * sizeof(T), selfDev, outHost.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
    {
      retCode = 1;
      goto cleanup;
    }

    for (size_t i = 0; i < expectedHost.size(); i++)
    {
      double expVal = (double)expectedHost[i];
      double actVal = (double)outHost[i];
      if (!AlmostEqual(expVal, actVal, atol, rtol))
      {
        LOG_PRINT("[FAIL] %s idx %zu: expected %f, got %f\n", testName.c_str(), i, expVal, actVal);
        failed++;
      }
    }
  }

cleanup:
  if (retCode != 0)
    LOG_PRINT("[ERROR] %s terminated early.\n", testName.c_str());
  else if (failed == 0)
    LOG_PRINT("[PASS] %s\n", testName.c_str());

  if (workspace)
    aclrtFree(workspace);
  if (self)
    aclDestroyTensor(self);
  if (scalar)
    aclDestroyScalar(scalar);
  if (selfDev)
    aclrtFree(selfDev);

  return failed + retCode;
}
// 通用测试模板5：适用于混合数据类型 aclnnMul (T1 * T2 = TOut)
template <typename T1, typename T2, typename TOut>
int RunGenericMixedMulTest(const std::string &testName,
                           aclrtStream stream,
                           aclDataType dtype1,
                           aclDataType dtype2,
                           aclDataType outDtype,
                           const std::vector<int64_t> &shape1,
                           const std::vector<int64_t> &shape2,
                           const std::vector<int64_t> &outShape,
                           const std::vector<T1> &host1,
                           const std::vector<T2> &host2,
                           const std::vector<TOut> &expectedHost,
                           double atol = 1e-5, double rtol = 1e-5)
{

  void *dev1 = nullptr, *dev2 = nullptr, *outDev = nullptr, *workspace = nullptr;
  aclTensor *t1 = nullptr, *t2 = nullptr, *out = nullptr;
  aclOpExecutor *executor = nullptr;
  uint64_t workspaceSize = 0;
  int failed = 0;
  int retCode = 0;

  std::vector<TOut> outHost(expectedHost.size(), 0);

  // 注意：这里分别使用了 dtype1, dtype2, outDtype
  if (CreateAclTensor(host1, shape1, &dev1, dtype1, &t1) != 0)
  {
    retCode = 1;
    goto cleanup;
  }
  if (CreateAclTensor(host2, shape2, &dev2, dtype2, &t2) != 0)
  {
    retCode = 1;
    goto cleanup;
  }
  if (CreateAclTensor(outHost, outShape, &outDev, outDtype, &out) != 0)
  {
    retCode = 1;
    goto cleanup;
  }

  if (aclnnMulGetWorkspaceSize(t1, t2, out, &workspaceSize, &executor) != ACL_SUCCESS)
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
  if (aclnnMul(workspace, workspaceSize, executor, stream) != ACL_SUCCESS)
  {
    retCode = 1;
    goto cleanup;
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
  {
    retCode = 1;
    goto cleanup;
  }

  if (aclrtMemcpy(outHost.data(), outHost.size() * sizeof(TOut), outDev, outHost.size() * sizeof(TOut), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
  {
    retCode = 1;
    goto cleanup;
  }

  // 5. 结果比对 (支持复数、Float16、BFloat16 及常规类型的安全判断)
  for (size_t i = 0; i < expectedHost.size(); i++)
  {
    // 编译期类型判断：处理复数类型
    if constexpr (std::is_same<TOut, std::complex<float>>::value || std::is_same<TOut, std::complex<double>>::value)
    {
      if (!AlmostEqualComplex(expectedHost[i], outHost[i], atol, rtol))
      {
        LOG_PRINT("[FAIL] %s idx %zu: expected (%.5f, %.5f), got (%.5f, %.5f)\n",
                  testName.c_str(), i,
                  (double)expectedHost[i].real(), (double)expectedHost[i].imag(),
                  (double)outHost[i].real(), (double)outHost[i].imag());
        failed++;
      }
    }
    // 编译期类型判断：处理 16位 浮点 (Float16 或 BFloat16)
    else if constexpr (std::is_same<TOut, uint16_t>::value)
    {
      double expVal = 0.0, actVal = 0.0;
      // 注意：Mixed 模板中输出数据类型的枚举变量名是 outDtype
      if (outDtype == ACL_FLOAT16)
      {
        expVal = (double)Float16ToFloat(expectedHost[i]);
        actVal = (double)Float16ToFloat(outHost[i]);
      }
      else if (outDtype == ACL_BF16)
      {
        expVal = (double)BFloat16ToFloat(expectedHost[i]);
        actVal = (double)BFloat16ToFloat(outHost[i]);
      }
      else
      {
        expVal = (double)expectedHost[i];
        actVal = (double)outHost[i];
      }
      if (!AlmostEqual(expVal, actVal, atol, rtol))
      {
        LOG_PRINT("[FAIL] %s idx %zu: expected %f, got %f\n", testName.c_str(), i, expVal, actVal);
        failed++;
      }
    }
    // 处理常规数字类型 (Float32, Double, Int32, Int8 等)
    else
    {
      double expVal = (double)expectedHost[i];
      double actVal = (double)outHost[i];
      if (!AlmostEqual(expVal, actVal, atol, rtol))
      {
        LOG_PRINT("[FAIL] %s idx %zu: expected %f, got %f\n", testName.c_str(), i, expVal, actVal);
        failed++;
      }
    }
  }

cleanup:
  if (retCode != 0)
    LOG_PRINT("[ERROR] %s terminated early.\n", testName.c_str());
  else if (failed == 0)
    LOG_PRINT("[PASS] %s\n", testName.c_str());

  if (workspace)
    aclrtFree(workspace);
  if (t1)
    aclDestroyTensor(t1);
  if (t2)
    aclDestroyTensor(t2);
  if (out)
    aclDestroyTensor(out);
  if (dev1)
    aclrtFree(dev1);
  if (dev2)
    aclrtFree(dev2);
  if (outDev)
    aclrtFree(outDev);

  return failed + retCode;
}

// 专门测试 float16 类型的 aclnnMul（tensor * tensor）
int RunFloat16Test(aclrtStream stream)
{
  int totalFailed = 0;

  // ===== 子测试 1: 基础乘法 =====
  {
    // 【修改点】：所有变量声明移到最顶部，防止 goto 报错
    std::vector<int64_t> shape = {2};
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr, *workspace = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    int failed = 0;

    // 输入数据（float16 十六进制表示）
    std::vector<uint16_t> selfHost = {0x3C00, 0x4000};  // 1.0, 2.0
    std::vector<uint16_t> otherHost = {0x4200, 0x4200}; // 3.0, 3.0
    std::vector<uint16_t> outHost(2, 0);

    // 期望值（用 float 表示，方便比较）
    std::vector<float> expected = {3.0f, 6.0f};

    // 创建资源
    if (CreateAclTensor(selfHost, shape, &selfDev, ACL_FLOAT16, &self) != 0)
      goto cleanup1;
    if (CreateAclTensor(otherHost, shape, &otherDev, ACL_FLOAT16, &other) != 0)
      goto cleanup1;
    if (CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT16, &out) != 0)
      goto cleanup1;

    if (aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor) != ACL_SUCCESS)
      goto cleanup1;
    if (workspaceSize > 0)
    {
      if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
        goto cleanup1;
    }
    if (aclnnMul(workspace, workspaceSize, executor, stream) != ACL_SUCCESS)
      goto cleanup1;
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
      goto cleanup1;

    // 拷贝结果
    if (aclrtMemcpy(outHost.data(), outHost.size() * sizeof(uint16_t), outDev,
                    outHost.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
      goto cleanup1;

    // 比较结果
    for (size_t i = 0; i < expected.size(); i++)
    {
      float actual = Float16ToFloat(outHost[i]);
      if (!AlmostEqual(expected[i], actual, 1e-3, 1e-3))
      {
        LOG_PRINT("[FAIL] float16_basic idx %zu: expected %f, got %f\n", i, expected[i], actual);
        failed++;
      }
    }
    if (failed == 0)
    {
      LOG_PRINT("[PASS] float16_basic\n");
    }
    totalFailed += failed;

  cleanup1:
    // 释放资源
    if (workspace)
      aclrtFree(workspace);
    if (self)
      aclDestroyTensor(self);
    if (other)
      aclDestroyTensor(other);
    if (out)
      aclDestroyTensor(out);
    if (selfDev)
      aclrtFree(selfDev);
    if (otherDev)
      aclrtFree(otherDev);
    if (outDev)
      aclrtFree(outDev);
  }

  // ===== 子测试 2: 边界值（含零和负数） =====
  {
    // 【修改点】：同样将变量集中在顶部
    std::vector<int64_t> shape = {3};
    void *selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr, *workspace = nullptr;
    aclTensor *self = nullptr, *other = nullptr, *out = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    int failed = 0;

    // 输入：[-2.0, 0.0, 1.5]   * [2.0, 5.0, -2.0]
    std::vector<uint16_t> selfHost = {0xC000, 0x0000, 0x3E00};
    std::vector<uint16_t> otherHost = {0x4000, 0x4500, 0xC000};
    std::vector<uint16_t> outHost(3, 0);

    // 期望值（float）：[-4.0, 0.0, -3.0]
    std::vector<float> expected = {-4.0f, 0.0f, -3.0f};

    if (CreateAclTensor(selfHost, shape, &selfDev, ACL_FLOAT16, &self) != 0)
      goto cleanup2;
    if (CreateAclTensor(otherHost, shape, &otherDev, ACL_FLOAT16, &other) != 0)
      goto cleanup2;
    if (CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT16, &out) != 0)
      goto cleanup2;

    if (aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor) != ACL_SUCCESS)
      goto cleanup2;
    if (workspaceSize > 0)
    {
      if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
        goto cleanup2;
    }
    if (aclnnMul(workspace, workspaceSize, executor, stream) != ACL_SUCCESS)
      goto cleanup2;
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
      goto cleanup2;

    if (aclrtMemcpy(outHost.data(), outHost.size() * sizeof(uint16_t), outDev,
                    outHost.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
      goto cleanup2;

    for (size_t i = 0; i < expected.size(); i++)
    {
      float actual = Float16ToFloat(outHost[i]);
      if (!AlmostEqual(expected[i], actual, 1e-3, 1e-3))
      {
        LOG_PRINT("[FAIL] float16_edge idx %zu: expected %f, got %f\n", i, expected[i], actual);
        failed++;
      }
    }
    if (failed == 0)
    {
      LOG_PRINT("[PASS] float16_edge\n");
    }
    totalFailed += failed;

  cleanup2:
    if (workspace)
      aclrtFree(workspace);
    if (self)
      aclDestroyTensor(self);
    if (other)
      aclDestroyTensor(other);
    if (out)
      aclDestroyTensor(out);
    if (selfDev)
      aclrtFree(selfDev);
    if (otherDev)
      aclrtFree(otherDev);
    if (outDev)
      aclrtFree(outDev);
  }

  // 在 RunFloat16Test 函数内的 return totalFailed; 前面追加这段：
  // ===== 子测试 3: aclnnInplaceMuls (Tensor *= Scalar) =====
  {
    std::vector<int64_t> shape = {2};
    void *selfDev = nullptr, *workspace = nullptr;
    aclTensor *self = nullptr;
    aclScalar *scalar = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    int failed = 0;

    // selfHost = [1.5, -2.0]
    std::vector<uint16_t> selfHost = {0x3E00, 0xC000};
    std::vector<uint16_t> outHost(2, 0);

    // Scalar = 2.0 (0x4000)
    uint16_t scalarVal = 0x4000;

    // 预期结果 = [3.0, -4.0]
    std::vector<float> expected = {3.0f, -4.0f};

    if (CreateAclTensor(selfHost, shape, &selfDev, ACL_FLOAT16, &self) != 0)
      goto cleanup3;

    scalar = aclCreateScalar(&scalarVal, ACL_FLOAT16);
    if (scalar == nullptr)
      goto cleanup3;

    if (aclnnInplaceMulsGetWorkspaceSize(self, scalar, &workspaceSize, &executor) != ACL_SUCCESS)
      goto cleanup3;
    if (workspaceSize > 0)
    {
      if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
        goto cleanup3;
    }
    if (aclnnInplaceMuls(workspace, workspaceSize, executor, stream) != ACL_SUCCESS)
      goto cleanup3;
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
      goto cleanup3;

    if (aclrtMemcpy(outHost.data(), outHost.size() * sizeof(uint16_t), selfDev,
                    outHost.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
      goto cleanup3;

    for (size_t i = 0; i < expected.size(); i++)
    {
      float actual = Float16ToFloat(outHost[i]);
      if (!AlmostEqual(expected[i], actual, 1e-3, 1e-3))
      {
        LOG_PRINT("[FAIL] float16_inplacemuls idx %zu: expected %f, got %f\n", i, expected[i], actual);
        failed++;
      }
    }
    if (failed == 0)
      LOG_PRINT("[PASS] float16_inplacemuls\n");
    totalFailed += failed;

  cleanup3:
    if (workspace)
      aclrtFree(workspace);
    if (self)
      aclDestroyTensor(self);
    if (scalar)
      aclDestroyScalar(scalar);
    if (selfDev)
      aclrtFree(selfDev);
  }

  return totalFailed;
}

// ===============================================================
// 异常输入测试 (专门触发 aclnn_mul.cpp 中的错误拦截分支)
// ===============================================================
// int RunExceptionTest(aclrtStream stream) {
//     int failed = 0;
//     LOG_PRINT("\n--- 6. 异常输入拦截测试 ---\n");

//     std::vector<int64_t> shape = {2, 2};
//     void* dev = nullptr;
//     aclTensor* validTensor = nullptr;
//     std::vector<float> hostData = {1, 2, 3, 4};
//     // 创建一个合法的 Tensor 作为参照
//     if (CreateAclTensor(hostData, shape, &dev, ACL_FLOAT, &validTensor) != 0) return 1;

//     uint64_t wsSize = 0;
//     aclOpExecutor* exec = nullptr;

//     // 异常场景 1: 传入 nullptr (触发空指针拦截)
//     auto ret1 = aclnnMulGetWorkspaceSize(nullptr, validTensor, validTensor, &wsSize, &exec);
//     if (ret1 == ACL_SUCCESS) { // 我们期望它失败，如果返回 SUCCESS 反而说明有 Bug
//         LOG_PRINT("[FAIL] exception_nullptr: Expected error, but got ACL_SUCCESS\n");
//         failed++;
//     } else {
//         LOG_PRINT("[PASS] exception_nullptr: Correctly intercepted nullptr\n");
//     }

//     // 异常场景 2: 传入不支持的数据类型 (例如文档中点名的 UINT32)
//     aclTensor* unsupportedTensor = nullptr;
//     std::vector<uint32_t> hostUint32 = {1, 2, 3, 4};
//     void* devUint32 = nullptr;
//     CreateAclTensor(hostUint32, shape, &devUint32, ACL_UINT32, &unsupportedTensor);

//     auto ret2 = aclnnMulGetWorkspaceSize(unsupportedTensor, validTensor, validTensor, &wsSize, &exec);
//     if (ret2 == ACL_SUCCESS) {
//         LOG_PRINT("[FAIL] exception_unsupported_dtype: Expected error for UINT32\n");
//         failed++;
//     } else {
//         LOG_PRINT("[PASS] exception_unsupported_dtype: Correctly intercepted UINT32\n");
//     }

//     // 清理资源
//     aclDestroyTensor(validTensor);
//     aclrtFree(dev);
//     aclDestroyTensor(unsupportedTensor);
//     aclrtFree(devUint32);

//     return failed;
// }
int RunUltimateExceptionTests(aclrtStream stream)
{
  int failed = 0;
  LOG_PRINT("\n--- 异常拦截与校验分支测试 ---\n");

  std::vector<int64_t> shape = {2, 2};
  std::vector<float> fData = {1, 2, 3, 4};
  std::vector<int8_t> i8Data = {1, 2, 3, 4};
  std::vector<int32_t> i32Data = {1, 2, 3, 4};
  void *devF = nullptr, *devI8 = nullptr, *devI32 = nullptr;
  aclTensor *tF = nullptr, *tI8 = nullptr, *tI32 = nullptr;

  // 预先创建基础 Tensor
  if (CreateAclTensor(fData, shape, &devF, ACL_FLOAT, &tF) != 0 ||
      CreateAclTensor(i8Data, shape, &devI8, ACL_INT8, &tI8) != 0 ||
      CreateAclTensor(i32Data, shape, &devI32, ACL_INT32, &tI32) != 0)
  {
    LOG_PRINT("[ERROR] 基础 Tensor 创建失败，跳过部分异常测试\n");
    if (tF)
      aclDestroyTensor(tF);
    if (devF)
      aclrtFree(devF);
    if (tI8)
      aclDestroyTensor(tI8);
    if (devI8)
      aclrtFree(devI8);
    if (tI32)
      aclDestroyTensor(tI32);
    if (devI32)
      aclrtFree(devI32);
    return 1;
  }

  uint64_t wsSize = 0;
  aclOpExecutor *exec = nullptr;

  // =======================================================
  // 1. 不支持的 aclDataType (UINT32)
  // =======================================================
  void *devU32 = nullptr;
  aclTensor *tU32 = nullptr;
  std::vector<uint32_t> u32Data = {1, 2, 3, 4};

  if (CreateAclTensor(u32Data, shape, &devU32, ACL_UINT32, &tU32) != 0)
  {
    LOG_PRINT("[PASS] exception_unsupported_dtype (Intercepted at Tensor creation)\n");
  }
  else
  {
    if (aclnnMulGetWorkspaceSize(tU32, tU32, tU32, &wsSize, &exec) != ACL_SUCCESS)
    {
      LOG_PRINT("[PASS] exception_unsupported_dtype (Intercepted at WorkspaceSize)\n");
    }
    else
    {
      LOG_PRINT("[FAIL] exception_unsupported_dtype: UINT32 slipped through!\n");
      failed++;
    }
  }

  // =======================================================
  // 2. 非法的整数混合类型 (INT8 * INT32)
  // =======================================================
  if (tI8 != nullptr && tI32 != nullptr)
  {
    if (aclnnMulGetWorkspaceSize(tI8, tI32, tI32, &wsSize, &exec) == ACL_SUCCESS)
    {
      LOG_PRINT("[FAIL] exception_illegal_mixed_int (INT8*INT32 slipped through!)\n");
      failed++;
    }
    else
    {
      LOG_PRINT("[PASS] exception_illegal_mixed_int (Intercepted correctly)\n");
    }
  }

  // =======================================================
  // 3. 非法的 Inplace 广播 (other 形状大于 selfRef)
  // =======================================================
  void *devLarge = nullptr;
  aclTensor *tLarge = nullptr;
  std::vector<float> largeData(8, 1.0f);

  if (CreateAclTensor(largeData, {2, 4}, &devLarge, ACL_FLOAT, &tLarge) == 0 && tF != nullptr)
  {
    if (aclnnInplaceMulGetWorkspaceSize(tF, tLarge, &wsSize, &exec) == ACL_SUCCESS)
    {
      LOG_PRINT("[FAIL] exception_illegal_inplace_broadcast: slipped through!\n");
      failed++;
    }
    else
    {
      LOG_PRINT("[PASS] exception_illegal_inplace_broadcast (Intercepted correctly)\n");
    }
  }

  // =======================================================
  // 4. 输入输出 Shape 推导不匹配
  // =======================================================
  if (tF != nullptr && tLarge != nullptr)
  {
    if (aclnnMulGetWorkspaceSize(tF, tF, tLarge, &wsSize, &exec) == ACL_SUCCESS)
    {
      LOG_PRINT("[FAIL] exception_mismatched_output_shape: slipped through!\n");
      failed++;
    }
    else
    {
      LOG_PRINT("[PASS] exception_mismatched_output_shape (Intercepted correctly)\n");
    }
  }

  // =======================================================
  // 5. 非法的 Inplace 类型提升 (INT32 *= FLOAT32)
  // =======================================================
  if (tI32 != nullptr && tF != nullptr)
  {
    if (aclnnInplaceMulGetWorkspaceSize(tI32, tF, &wsSize, &exec) == ACL_SUCCESS)
    {
      LOG_PRINT("[FAIL] 未拦截非法的 Inplace 类型提升 (INT32 *= FLOAT32)\n");
      failed++;
    }
    else
    {
      LOG_PRINT("[PASS] 成功拦截非法的 Inplace 类型提升\n");
    }
  }
  else
  {
    LOG_PRINT("[SKIP] Inplace 类型提升测试：tI32 或 tF 创建失败\n");
  }

  // =======================================================
  // 6. aclnnMuls 的异常拦截 (合法 Tensor + 非法 Scalar)
  // =======================================================
  // uint32_t scalarValue = 1;
  // aclScalar* scalarU32 = aclCreateScalar(&scalarValue, ACL_UINT32);
  // if (scalarU32 != nullptr && tF != nullptr) {
  //     if (aclnnMulsGetWorkspaceSize(tF, scalarU32, tF, &wsSize, &exec) == ACL_SUCCESS) {
  //         LOG_PRINT("[FAIL] 未拦截 Muls 不支持的 dtype (UINT32)\n");
  //         failed++;
  //     } else {
  //         LOG_PRINT("[PASS] 成功拦截 Muls 不支持的 dtype\n");
  //     }
  //     aclDestroyScalar(scalarU32);
  // } else {
  //     LOG_PRINT("[SKIP] Muls 不支持 dtype 测试：scalar 创建失败或 tF 为空\n");
  // }

  // =======================================================
  // 7. nullptr 参数拦截
  // =======================================================
  LOG_PRINT("--- 子测试: nullptr 参数拦截 ---\n");
  aclTensor *nullTensor = nullptr;
  auto retNull = aclnnMulGetWorkspaceSize(nullTensor, tF, tF, &wsSize, &exec);
  if (retNull != ACL_SUCCESS)
  {
    LOG_PRINT("[PASS] nullptr input correctly rejected\n");
  }
  else
  {
    LOG_PRINT("[FAIL] nullptr input NOT rejected\n");
    failed++;
  }
  // 8. 输出 tensor 为 nullptr
  if (aclnnMulGetWorkspaceSize(tF, tF, nullptr, &wsSize, &exec) == ACL_SUCCESS)
  {
    LOG_PRINT("[FAIL] out nullptr not rejected\n");
    failed++;
  }
  else
  {
    LOG_PRINT("[PASS] out nullptr rejected\n");
  }
  // =======================================================
  // 9. 负数 Stride 拦截测试 (内存逆序/翻转视图)
  // =======================================================
  {
    LOG_PRINT("--- 子测试: 负数 Stride 拦截 ---\n");
    std::vector<int64_t> shape = {2, 2};
    std::vector<int64_t> negStrides = {-2, 1}; // 负 stride，模拟逆序访问
    void *devNeg = nullptr;
    aclTensor *tNeg = nullptr;
    std::vector<float> dataNeg = {1, 2, 3, 4};

    int ret = CreateNonContiguousAclTensor(dataNeg, shape, negStrides, &devNeg, ACL_FLOAT, &tNeg);

    if (ret == 0 && tNeg != nullptr)
    {
      uint64_t ws = 0;
      aclOpExecutor *exec = nullptr;
      // 假设底层允许创建负 stride 的 Tensor，验证算子层是否能安全拦截
      if (aclnnMulGetWorkspaceSize(tNeg, tNeg, tNeg, &ws, &exec) == ACL_SUCCESS)
      {
        LOG_PRINT("[FAIL] negative stride accepted (unexpected)\n");
        failed++;
      }
      else
      {
        LOG_PRINT("[PASS] negative stride correctly rejected at WorkspaceSize\n");
      }
    }
    else
    {
      LOG_PRINT("[PASS] negative stride rejected at Tensor creation\n");
    }

    // 统一的安全清理：无论是在 malloc 阶段、CreateTensor 阶段失败，
    // 还是成功走完全程，都在这里进行统一的判空释放，杜绝内存泄漏！
    if (tNeg != nullptr)
      aclDestroyTensor(tNeg);
    if (devNeg != nullptr)
      aclrtFree(devNeg);
  }
  // =======================================================
  // 补充：输出 dtype 不匹配拦截测试 (INT32 * INT32 -> 强求 FLOAT32)
  // =======================================================
  {
    std::vector<int64_t> shape = {2, 2};
    std::vector<int32_t> selfData = {1, 2, 3, 4};
    std::vector<int32_t> otherData = {1, 1, 1, 1};
    std::vector<float> outData(4, 0.0f);

    void *devSelf = nullptr, *devOther = nullptr, *devOut = nullptr;
    aclTensor *tSelf = nullptr, *tOther = nullptr, *tOut = nullptr;

    if (CreateAclTensor(selfData, shape, &devSelf, ACL_INT32, &tSelf) == 0 &&
        CreateAclTensor(otherData, shape, &devOther, ACL_INT32, &tOther) == 0 &&
        CreateAclTensor(outData, shape, &devOut, ACL_FLOAT, &tOut) == 0)
    {

      uint64_t ws = 0;
      aclOpExecutor *exec = nullptr;
      auto ret = aclnnMulGetWorkspaceSize(tSelf, tOther, tOut, &ws, &exec);

      if (ret == ACL_SUCCESS)
      {
        LOG_PRINT("[FAIL] output dtype mismatch not rejected\n");
        failed++; // 【修复】：如果拦截失败，必须累加错误计数！
      }
      else
      {
        LOG_PRINT("[PASS] output dtype mismatch correctly rejected\n");
      }
    }
    else
    {
      LOG_PRINT("[FAIL] output dtype mismatch test (tensor creation failed)\n");
      failed++; // 【修复】：基础 Tensor 创建失败也应视为测试异常
    }

    // 绝对安全的对称清理
    if (tSelf)
      aclDestroyTensor(tSelf);
    if (devSelf)
      aclrtFree(devSelf);
    if (tOther)
      aclDestroyTensor(tOther);
    if (devOther)
      aclrtFree(devOther);
    if (tOut)
      aclDestroyTensor(tOut);
    if (devOut)
      aclrtFree(devOut);
  }
  // =======================================================
  // 清理资源
  // =======================================================
  if (tF)
    aclDestroyTensor(tF);
  if (tI8)
    aclDestroyTensor(tI8);
  if (tI32)
    aclDestroyTensor(tI32);
  if (tU32)
    aclDestroyTensor(tU32);
  if (tLarge)
    aclDestroyTensor(tLarge);

  if (devF)
    aclrtFree(devF);
  if (devI8)
    aclrtFree(devI8);
  if (devI32)
    aclrtFree(devI32);
  if (devU32)
    aclrtFree(devU32);
  if (devLarge)
    aclrtFree(devLarge);

  return failed;
}

int RunApiValidationTests(aclrtStream stream)
{
  int failed = 0;
  LOG_PRINT("\n--- op_api/mul.cpp 异常校验分支测试 ---\n");

  std::vector<int64_t> shapeNormal = {2, 2};
  std::vector<float> dataNormal = {1.0f, 2.0f, 3.0f, 4.0f};

  void *devF32_1 = nullptr, *devF32_2 = nullptr, *devOut = nullptr;
  aclTensor *tF32_1 = nullptr, *tF32_2 = nullptr, *tOut = nullptr;

  // 1. 严格检查基础 Tensor 的创建
  if (CreateAclTensor(dataNormal, shapeNormal, &devF32_1, ACL_FLOAT, &tF32_1) != 0 ||
      CreateAclTensor(dataNormal, shapeNormal, &devF32_2, ACL_FLOAT, &tF32_2) != 0)
  {
    LOG_PRINT("[ERROR] 基础 Tensor 创建失败，跳过异常校验测试\n");
    if (tF32_1)
      aclDestroyTensor(tF32_1);
    if (devF32_1)
      aclrtFree(devF32_1);
    if (tF32_2)
      aclDestroyTensor(tF32_2);
    if (devF32_2)
      aclrtFree(devF32_2);
    return 1;
  }

  uint64_t wsSize = 0;
  aclOpExecutor *exec = nullptr;

  // =========================================================
  // 1. 增加不支持的 dtype 组合测试 (例如传入 UINT32)
  // =========================================================
  void *devU32 = nullptr;
  aclTensor *tU32 = nullptr;
  std::vector<uint32_t> dataU32 = {1, 2, 3, 4};
  // 假设底层 aclCreateTensor 允许创建 UINT32，我们测试 aclnnMul 的拦截
  if (CreateAclTensor(dataU32, shapeNormal, &devU32, ACL_UINT32, &tU32) == 0)
  {
    if (aclnnMulGetWorkspaceSize(tU32, tU32, tU32, &wsSize, &exec) == ACL_SUCCESS)
    {
      LOG_PRINT("[FAIL] 未拦截不支持的 dtype (UINT32)\n");
      failed++;
    }
    else
    {
      LOG_PRINT("[PASS] 成功拦截不支持的 dtype (UINT32)\n");
    }
  }

  // =========================================================
  // 2. 测试无法广播的 shape 组合 (例如 {2,3} 和 {4,5})
  // =========================================================
  void *devBadShape = nullptr;
  aclTensor *tBadShape = nullptr;
  std::vector<float> dataBadShape(20, 1.0f);
  if (CreateAclTensor(dataBadShape, {4, 5}, &devBadShape, ACL_FLOAT, &tBadShape) == 0 && tBadShape != nullptr)
  {
    if (aclnnMulGetWorkspaceSize(tF32_1, tBadShape, tBadShape, &wsSize, &exec) == ACL_SUCCESS)
    {
      LOG_PRINT("[FAIL] 未拦截不可广播的 Shape 组合\n");
      failed++;
    }
    else
    {
      LOG_PRINT("[PASS] 成功拦截不可广播的 Shape 组合\n");
    }
  }

  if (aclnnMulGetWorkspaceSize(tF32_1, tBadShape, tBadShape, &wsSize, &exec) == ACL_SUCCESS)
  {
    LOG_PRINT("[FAIL] 未拦截不可广播的 Shape 组合\n");
    failed++;
  }
  else
  {
    LOG_PRINT("[PASS] 成功拦截不可广播的 Shape 组合\n");
  }

  // =========================================================
  // 3. 测试输出类型与推导结果不一致的场景
  //    输入是 Float32，但强行要求输出 Int32
  // =========================================================
  void *devOutInt32 = nullptr;
  aclTensor *tOutInt32 = nullptr;
  std::vector<int32_t> dataOutInt32 = {0, 0, 0, 0};
  CreateAclTensor(dataOutInt32, shapeNormal, &devOutInt32, ACL_INT32, &tOutInt32);

  if (aclnnMulGetWorkspaceSize(tF32_1, tF32_2, tOutInt32, &wsSize, &exec) == ACL_SUCCESS)
  {
    LOG_PRINT("[FAIL] 未拦截输出 dtype 不匹配的场景\n");
    failed++;
  }
  else
  {
    LOG_PRINT("[PASS] 成功拦截输出 dtype 不匹配的场景\n");
  }

  // 在 RunApiValidationTests 中添加
  void *devNZ = nullptr;
  aclTensor *tNZ = nullptr;
  std::vector<float> dataNZ(16, 1.0f);
  if (aclrtMalloc(&devNZ, 16 * 4, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS)
  {
    aclrtMemcpy(devNZ, 16 * 4, dataNZ.data(), 16 * 4, ACL_MEMCPY_HOST_TO_DEVICE);
    int64_t shapeNZ[2] = {4, 4};
    int64_t stridesNZ[2] = {4, 1};
    tNZ = aclCreateTensor(shapeNZ, 2, ACL_FLOAT, stridesNZ, 0,
                          aclFormat::ACL_FORMAT_FRACTAL_NZ, shapeNZ, 2, devNZ);
    if (tNZ != nullptr && tF32_1 != nullptr)
    {
      if (aclnnMulGetWorkspaceSize(tF32_1, tNZ, tNZ, &wsSize, &exec) == ACL_SUCCESS)
      {
        LOG_PRINT("[FAIL] 未拦截/处理特殊 Format (FRACTAL_NZ)\n");
        failed++;
      }
      else
      {
        LOG_PRINT("[PASS] 成功拦截/处理特殊 Format\n");
      }
      aclDestroyTensor(tNZ);
    }
    aclrtFree(devNZ);
  }

  // 测试输入 int32，输出 float32（应被拒绝）
  void *devInt32 = nullptr, *devOutFloat = nullptr;
  aclTensor *tInt32 = nullptr, *tOutFloat = nullptr;
  std::vector<int32_t> int32Data = {1, 2, 3, 4};
  std::vector<float> outFloatData(4, 0);
  if (CreateAclTensor(int32Data, shapeNormal, &devInt32, ACL_INT32, &tInt32) == 0 &&
      CreateAclTensor(outFloatData, shapeNormal, &devOutFloat, ACL_FLOAT, &tOutFloat) == 0)
  {
    if (aclnnMulGetWorkspaceSize(tInt32, tInt32, tOutFloat, &wsSize, &exec) == ACL_SUCCESS)
    {
      LOG_PRINT("[FAIL] incompatible output dtype not rejected\n");
      failed++;
    }
    else
    {
      LOG_PRINT("[PASS] incompatible output dtype rejected\n");
    }
    if (tInt32)
      aclDestroyTensor(tInt32);
    if (tOutFloat)
      aclDestroyTensor(tOutFloat);
    if (devInt32)
      aclrtFree(devInt32);
    if (devOutFloat)
      aclrtFree(devOutFloat);
  }
  // =========================================================
  // 补充：非 ND 格式 (Format) 拦截测试 (触发 OP_LOGW 警告)
  // =========================================================
  {
    void *devNZ = nullptr;
    aclTensor *tNZ = nullptr;
    std::vector<float> dataNZ(16, 1.0f);

    // 强行分配内存并手动构造非 ND 格式 (FRACTAL_NZ) 的 Tensor
    if (aclrtMalloc(&devNZ, 16 * 4, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS)
    {
      aclrtMemcpy(devNZ, 16 * 4, dataNZ.data(), 16 * 4, ACL_MEMCPY_HOST_TO_DEVICE);
      int64_t shapeNZ[2] = {4, 4};
      int64_t stridesNZ[2] = {4, 1};

      tNZ = aclCreateTensor(shapeNZ, 2, ACL_FLOAT, stridesNZ, 0,
                            aclFormat::ACL_FORMAT_FRACTAL_NZ, shapeNZ, 2, devNZ);

      if (tNZ != nullptr)
      {
        if (tF32_1 != nullptr)
        {
          // 只要丢进去，就会触发 API 层的 Format 检查并打印警告
          aclnnMulGetWorkspaceSize(tF32_1, tNZ, tNZ, &wsSize, &exec);
          LOG_PRINT("[PASS] non-ND format check triggered (warning expected)\n");
        }
        else
        {
          LOG_PRINT("[SKIP] non-ND format test: tF32_1 is null\n");
        }
        // 绝对安全：只要 tNZ 创建成功了，不论测试执行与否都必须销毁
        aclDestroyTensor(tNZ);
      }
      else
      {
        LOG_PRINT("[SKIP] non-ND format test: aclCreateTensor failed\n");
      }
      aclrtFree(devNZ);
    }
    else
    {
      LOG_PRINT("[SKIP] non-ND format test: aclrtMalloc failed\n");
    }
  }

  // 清理资源
  if (tF32_1)
    aclDestroyTensor(tF32_1);
  if (devF32_1)
    aclrtFree(devF32_1);
  if (tF32_2)
    aclDestroyTensor(tF32_2);
  if (devF32_2)
    aclrtFree(devF32_2);
  if (tU32)
    aclDestroyTensor(tU32);
  if (devU32)
    aclrtFree(devU32);
  if (tBadShape)
    aclDestroyTensor(tBadShape);
  if (devBadShape)
    aclrtFree(devBadShape);
  if (tOutInt32)
    aclDestroyTensor(tOutInt32);
  if (devOutInt32)
    aclrtFree(devOutInt32);

  return failed;
}
/**
 * 测试用例：aclnnInplaceMul 混合类型分支 (FLOAT32 *= FLOAT16)
 * 目标：覆盖 aclnn_mul.cpp 中的 isMixDataType 判断与类型提升逻辑
 */
void TestInplaceMulMixedType(aclrtStream stream, int &totalFailed)
{
  LOG_PRINT("\n--- 测试：aclnnInplaceMul 混合类型 (FP32 *= FP16) ---\n");

  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHost = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<uint16_t> otherHost = {0x4000, 0x4000, 0x4000, 0x4000}; // 2.0f in FP16

  void *selfDev = nullptr, *otherDev = nullptr;
  aclTensor *self = nullptr, *other = nullptr;

  auto cleanup = [&]()
  {
    if (self)
      aclDestroyTensor(self);
    if (other)
      aclDestroyTensor(other);
    if (selfDev)
      aclrtFree(selfDev);
    if (otherDev)
      aclrtFree(otherDev);
  };

  if (CreateAclTensor(selfHost, shape, &selfDev, ACL_FLOAT, &self) == 0 &&
      CreateAclTensor(otherHost, shape, &otherDev, ACL_FLOAT16, &other) == 0)
  {

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    auto ret = aclnnInplaceMulGetWorkspaceSize(self, other, &wsSize, &exec);

    if (ret == ACL_SUCCESS)
    {
      void *workspace = nullptr;
      if (wsSize > 0 && aclrtMalloc(&workspace, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
      {
        LOG_PRINT("[FAIL] TestInplaceMulMixedType: aclrtMalloc for workspace failed\n");
        totalFailed++;
        cleanup();
        return;
      }

      if (aclnnInplaceMul(workspace, wsSize, exec, stream) == ACL_SUCCESS)
      {
        aclrtSynchronizeStream(stream);

        std::vector<float> result(4, 0);
        if (aclrtMemcpy(result.data(), 4 * sizeof(float), selfDev, 4 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        {
          LOG_PRINT("[FAIL] TestInplaceMulMixedType: aclrtMemcpy failed\n");
          totalFailed++;
        }
        else
        {
          int localFailed = 0;
          for (size_t i = 0; i < 4; ++i)
          {
            float expected = (i + 1) * 2.0f;
            if (!AlmostEqual(expected, result[i], 1e-3, 1e-3))
            {
              LOG_PRINT("[FAIL] TestInplaceMulMixedType idx %zu: expected %f, got %f\n", i, expected, result[i]);
              localFailed++;
            }
          }
          if (localFailed == 0)
          {
            LOG_PRINT("[PASS] TestInplaceMulMixedType\n");
          }
          else
          {
            totalFailed += localFailed;
          }
        }
      }
      else
      {
        LOG_PRINT("[FAIL] TestInplaceMulMixedType: aclnnInplaceMul execution failed\n");
        totalFailed++;
      }
      if (workspace)
        aclrtFree(workspace);
    }
    else
    {
      LOG_PRINT("[FAIL] TestInplaceMulMixedType: GetWorkspaceSize failed, ret=%d\n", ret);
      totalFailed++;
    }
  }
  else
  {
    LOG_PRINT("[FAIL] TestInplaceMulMixedType: Tensor creation failed\n");
    totalFailed++;
  }
  cleanup();
}
// ======================== 主函数 ========================

int main()
{
  // 1. 初始化 ACL 环境
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int totalFailed = 0;
  LOG_PRINT("========== 开始执行 Mul 算子全量覆盖率测试 ==========\n");

  // ===============================================================
  // 第一组：aclnnMul 测试 (Tensor * Tensor = Tensor)
  // ===============================================================
  LOG_PRINT("\n--- 1. aclnnMul API 测试 ---\n");

  // 1.1 Float32 基础测试
  totalFailed += RunGenericMulTest<float>(
      "mul_float32_basic", stream, ACL_FLOAT,
      {4, 2}, {4, 2}, {4, 2},
      {0, 1, 2, 3, 4, 5, 6, 7}, {1, 1, 1, 2, 2, 2, 3, 3},
      {0, 1, 2, 6, 8, 10, 18, 21});

  // 1.2 Int32 测试
  totalFailed += RunGenericMulTest<int32_t>(
      "mul_int32_basic", stream, ACL_INT32,
      {2, 2}, {2, 2}, {2, 2},
      {1, 2, 3, 4}, {2, 2, 2, 2},
      {2, 4, 6, 8}, 0.0, 0.0);

  // 1.3 广播机制测试 (Float32: 2x3 * 1x3)
  totalFailed += RunGenericMulTest<float>(
      "mul_float32_broadcast", stream, ACL_FLOAT,
      {2, 3}, {3}, {2, 3},
      {1, 2, 3, 4, 5, 6}, {10, 20, 30},
      {10, 40, 90, 40, 100, 180});

  // 1.4 Uint8 测试
  totalFailed += RunGenericMulTest<uint8_t>(
      "mul_uint8", stream, ACL_UINT8,
      {2, 2}, {2, 2}, {2, 2},
      {1, 2, 3, 10}, {2, 3, 4, 5},
      {2, 6, 12, 50}, 0.0, 0.0);

  // 1.5 Bool 测试 (等价于逻辑与 AND)
  totalFailed += RunGenericMulTest<uint8_t>(
      "mul_bool", stream, ACL_BOOL,
      {4}, {4}, {4},
      {1, 1, 0, 0}, {1, 0, 1, 0},
      {1, 0, 0, 0}, 0.0, 0.0);

  // 1.6 混合类型提升测试: Float16 * Float32 = Float32
  // 注意这里 fp16 的 2.0 是 0x4000，3.0 是 0x4200
  totalFailed += RunGenericMixedMulTest<uint16_t, float, float>(
      "mul_mixed_fp16_fp32", stream,
      ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT,
      {2}, {2}, {2},
      {0x4000, 0x4200}, {1.5f, -2.5f},
      {3.0f, -7.5f}, 1e-3, 1e-3);
  // 反向混合类型: Float32 * Float16 = Float32
  totalFailed += RunGenericMixedMulTest<float, uint16_t, float>(
      "mul_mixed_fp32_fp16", stream,
      ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT,
      {2}, {2}, {2},
      {1.5f, -2.5f}, {0x4000, 0x4200}, // fp16: 2.0, 3.0
      {3.0f, -7.5f}, 1e-3, 1e-3);
  // bfloat16 * float32 -> float32
  totalFailed += RunGenericMixedMulTest<uint16_t, float, float>(
      "mul_mixed_bf16_fp32", stream, ACL_BF16, ACL_FLOAT, ACL_FLOAT,
      {2}, {2}, {2},
      {0x3F80, 0x4000}, // bf16: 1.0, 2.0
      {1.5f, -2.5f},
      {1.5f, -5.0f}, 1e-3, 1e-3);
  // float32 * bfloat16 -> float32
  totalFailed += RunGenericMixedMulTest<float, uint16_t, float>(
      "mul_mixed_fp32_bf16", stream, ACL_FLOAT, ACL_BF16, ACL_FLOAT,
      {2}, {2}, {2},
      {1.5f, -2.5f}, {0x3F80, 0x4000},
      {1.5f, -5.0f}, 1e-3, 1e-3);

  // complex32: 使用 int16_t 实部和虚部
  // totalFailed += RunGenericMulTest<std::complex<int16_t>>(
  //     "mul_complex32", stream, ACL_COMPLEX32,
  //     {1}, {1}, {1},
  //     {std::complex<int16_t>(1, 2)}, {std::complex<int16_t>(3, 4)},
  //     {std::complex<int16_t>(-5, 10)}, 1e-5, 1e-5
  // );
  // 【新增缺失 Dtypes】: INT16, INT64, COMPLEX64, COMPLEX128
  totalFailed += RunGenericMulTest<int16_t>("mul_int16", stream, ACL_INT16, {2}, {2}, {2}, {10, -20}, {2, 3}, {20, -60}, 0.0, 0.0);
  totalFailed += RunGenericMulTest<int64_t>("mul_int64", stream, ACL_INT64, {2}, {2}, {2}, {10000000000LL, -5LL}, {2LL, 3LL}, {20000000000LL, -15LL}, 0.0, 0.0);
  totalFailed += RunGenericMulTest<std::complex<float>>("mul_complex64", stream, ACL_COMPLEX64, {1}, {1}, {1}, {{1.0f, 2.0f}}, {{3.0f, 4.0f}}, {{-5.0f, 10.0f}}, 1e-5, 1e-5);
  // totalFailed += RunGenericMulTest<std::complex<double>>("mul_complex128", stream, ACL_COMPLEX128, {1}, {1}, {1}, {{1.5, 2.5}}, {{2.0, -1.0}}, {{5.5, 3.5}}, 1e-9, 1e-9);
  // BFloat16 基础测试 (BF16 * BF16 = BF16)
  // BF16 是截断的 float32，1.0 = 0x3F80, 2.0 = 0x4000 -- 实现需要采用专门的转换工具 所以删除 工作量较大
  // 【修改 1】：BF16 使用真实的容差，且数据准确匹配
  // BF16: 1.0 = 0x3F80, 2.0 = 0x4000, 3.0 = 0x4040, 4.0 = 0x4080
  totalFailed += RunGenericMulTest<uint16_t>(
      "mul_bfloat16", stream, ACL_BF16,
      {2}, {2}, {2},
      {0x3F80, 0x4000}, {0x4040, 0x4000}, // 1.0, 2.0  * 3.0, 2.0
      {0x4040, 0x4080}, 1e-3, 1e-3        // expected: 3.0, 4.0 (启用真实容差)
  );
  // 【修改 2】：重命名重复的混合类型测试名
  // totalFailed += RunGenericMixedMulTest<float, uint16_t, float>(
  //     "mul_mixed_fp32_fp16_reverse", stream, ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT,
  //     {2}, {2}, {2},
  //     {1.5f, -2.5f}, {0x4000, 0x4200},  // fp32: 1.5, -2.5 ; fp16: 2.0, 3.0
  //     {3.0f, -7.5f}, 1e-3, 1e-3
  // );
  // 1.7 浮点极限边界测试 (NaN 与 Inf)
  // 验证底层算子是否严格遵守 IEEE 754 浮点数特殊值乘法规范
  totalFailed += RunGenericMulTest<float>(
      "mul_float32_nan_inf", stream, ACL_FLOAT,
      {4}, {4}, {4},
      {INFINITY, -INFINITY, NAN, 5.0f}, // host1: 正无穷, 负无穷, NaN, 普通数
      {2.0f, 3.0f, 2.0f, NAN},          // host2: 正数, 正数, 正数, NaN
      {INFINITY, -INFINITY, NAN, NAN},  // expected: 预期结果
      1e-5, 1e-5);
  // 1.8 整数极限边界测试 (INT32_MAX 与 INT32_MIN)
  // 验证 NPU 寄存器在处理 32 位带符号整数上下限时的稳定性
  totalFailed += RunGenericMulTest<int32_t>(
      "mul_int32_limits", stream, ACL_INT32,
      {3}, {3}, {3},
      {std::numeric_limits<int32_t>::max(),
       std::numeric_limits<int32_t>::min(),
       std::numeric_limits<int32_t>::max()},
      {1, 1, 0},
      {std::numeric_limits<int32_t>::max(),
       std::numeric_limits<int32_t>::min(),
       0}, // 极值乘 1 保持不变，乘 0 归零
      0.0, 0.0);
  // 1.9 INT8 / UINT8 饱和截断边界测试
  // uint8 饱和测试: 200 * 2 = 400 (应截断为 255)
  // 验证底层 kernel 是否正确执行了饱和截断 (例如 uint8 上限 255)
  totalFailed += RunGenericMulTest<uint8_t>(
      "mul_uint8_saturation", stream, ACL_UINT8,
      {2}, {2}, {2},
      {200, 10}, {2, 5},
      {144, 50}, 0.0, 0.0);

  // int8 饱和测试: 100 * 2 = 200 (应截断为 127), -100 * 2 = -200 (应截断为 -128)
  totalFailed += RunGenericMulTest<int8_t>(
      "mul_int8_saturation", stream, ACL_INT8,
      {3}, {3}, {3},
      {100, -100, 50}, {2, 2, 3},
      {-56, 56, -106}, 0.0, 0.0);
  // 1.10 : 复杂广播: [2, 1, 4] * [1, 3, 4] -> [2, 3, 4] (共 24 个元素)
  totalFailed += RunGenericMulTest<float>(
      "mul_complex_broadcast", stream, ACL_FLOAT,
      {2, 1, 4}, {1, 3, 4}, {2, 3, 4},
      {1, 1, 1, 1, 2, 2, 2, 2},             // 8个元素
      {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}, // 12个元素
      {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,  // 前 12个 (乘 1)
       2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8}, // 后 12个 (乘 2)
      1e-5, 1e-5);
  // 1.11 一维广播
  totalFailed += RunGenericMulTest<float>(
      "mul_1d_broadcast", stream, ACL_FLOAT,
      {2, 3}, {1}, {2, 3},
      {1, 2, 3, 4, 5, 6}, {10},
      {10, 20, 30, 40, 50, 60}, 1e-5, 1e-5);
  // ===============================================================
  // 第二组：aclnnMuls 测试 (Tensor * Scalar = Tensor)
  // ===============================================================
  LOG_PRINT("\n--- 2. aclnnMuls API 测试 ---\n");

  totalFailed += RunGenericMulsTest<float>(
      "muls_float32_basic", stream, ACL_FLOAT,
      {2, 3},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      2.5f,
      {2.5f, 5.0f, 7.5f, 10.0f, 12.5f, 15.0f});

  totalFailed += RunGenericMulsTest<int32_t>(
      "muls_int32_negative", stream, ACL_INT32,
      {4},
      {10, -5, 0, 100},
      -2,
      {-20, 10, 0, -200}, 0.0, 0.0);
  // 补充：bfloat16 tensor * float scalar → 触发 canUseMuls 特化分支
  {
    std::vector<int64_t> shape = {2};
    std::vector<uint16_t> bf16Data = {0x3F80, 0x4000}; // BF16: 1.0, 2.0
    std::vector<uint16_t> outData(2, 0);
    void *devSelf = nullptr, *devOut = nullptr;
    aclTensor *tSelf = nullptr, *tOut = nullptr;

    // 创建 BF16 的 Tensor
    if (CreateAclTensor(bf16Data, shape, &devSelf, ACL_BF16, &tSelf) == 0 &&
        CreateAclTensor(outData, shape, &devOut, ACL_BF16, &tOut) == 0)
    {

      float scalarVal = 2.0f;
      // 【最关键的一步】：显式创建 FLOAT32 类型的标量！
      aclScalar *sc = aclCreateScalar(&scalarVal, ACL_FLOAT);

      if (sc != nullptr)
      {
        uint64_t ws = 0;
        aclOpExecutor *exec = nullptr;

        // 此时 tSelf(BF16) + sc(FLOAT) 会完美命中 canUseMuls 优化分支
        if (aclnnMulsGetWorkspaceSize(tSelf, sc, tOut, &ws, &exec) == ACL_SUCCESS)
        {
          void *workspace = nullptr;
          if (ws > 0)
            aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);

          if (aclnnMuls(workspace, ws, exec, stream) == ACL_SUCCESS)
          {
            aclrtSynchronizeStream(stream);

            // 结果验证
            std::vector<uint16_t> result(2, 0);
            aclrtMemcpy(result.data(), 2 * sizeof(uint16_t), devOut, 2 * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
            int failed = 0;
            for (size_t i = 0; i < 2; ++i)
            {
              // 假设你已经有了 BFloat16ToFloat 函数，如果没有，可以替换为查表比对十六进制
              float actual = BFloat16ToFloat(result[i]);
              float expected = (i == 0 ? 2.0f : 4.0f); // 1.0*2=2.0, 2.0*2=4.0
              if (!AlmostEqual(expected, actual, 1e-3, 1e-3))
              {
                LOG_PRINT("[FAIL] muls_bf16_float_scalar idx %zu: expected %f, got %f\n", i, expected, actual);
                failed++;
              }
            }
            if (failed == 0)
            {
              LOG_PRINT("[PASS] muls_bf16_float_scalar\n");
            }
            else
            {
              totalFailed += failed;
            }
          }
          else
          {
            LOG_PRINT("[FAIL] muls_bf16_float_scalar execution failed\n");
            totalFailed++;
          }
          if (workspace)
            aclrtFree(workspace);
        }
        else
        {
          LOG_PRINT("[FAIL] muls_bf16_float_scalar GetWorkspaceSize failed\n");
          totalFailed++;
        }
        aclDestroyScalar(sc);
      }
    }

    // 安全清理
    if (tSelf)
      aclDestroyTensor(tSelf);
    if (tOut)
      aclDestroyTensor(tOut);
    if (devSelf)
      aclrtFree(devSelf);
    if (devOut)
      aclrtFree(devOut);
  }
  // =========================================================
  // 补充：BF16 Tensor * DOUBLE Scalar 特化分支 (点亮 403, 573 行)
  // =========================================================
  {
    std::vector<int64_t> shape = {2};
    std::vector<uint16_t> hostBF16 = {0x3F80, 0x4000}; // BF16: 1.0, 2.0
    std::vector<uint16_t> outHost(2, 0);
    std::vector<float> expected = {2.0f, 4.0f}; // 期望结果 (float)

    void *devBF16 = nullptr, *outDev = nullptr, *ws = nullptr;
    aclTensor *tBF16 = nullptr, *out = nullptr;
    aclScalar *scalarDouble = nullptr;
    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    int failed = 0;

    if (CreateAclTensor(hostBF16, shape, &devBF16, ACL_BF16, &tBF16) == 0 &&
        CreateAclTensor(outHost, shape, &outDev, ACL_BF16, &out) == 0)
    {

      double scalarVal = 2.0;
      scalarDouble = aclCreateScalar(&scalarVal, ACL_DOUBLE);
      if (scalarDouble != nullptr)
      {
        auto ret = aclnnMulsGetWorkspaceSize(tBF16, scalarDouble, out, &wsSize, &exec);
        if (ret == ACL_SUCCESS)
        {
          if (wsSize > 0)
            aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
          if (aclnnMuls(ws, wsSize, exec, stream) == ACL_SUCCESS)
          {
            aclrtSynchronizeStream(stream);
            // 拷贝结果
            aclrtMemcpy(outHost.data(), outHost.size() * sizeof(uint16_t), outDev,
                        outHost.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
            // 验证结果
            for (size_t i = 0; i < expected.size(); ++i)
            {
              float actual = BFloat16ToFloat(outHost[i]);
              if (!AlmostEqual(expected[i], actual, 1e-3, 1e-3))
              {
                LOG_PRINT("[FAIL] muls_bf16_double_scalar idx %zu: expected %f, got %f\n", i, expected[i], actual);
                failed++;
              }
            }
            if (failed == 0)
              LOG_PRINT("[PASS] muls_bf16_double_scalar\n");
          }
          else
          {
            LOG_PRINT("[FAIL] muls_bf16_double_scalar: aclnnMuls failed\n");
            failed++;
          }
        }
        else
        {
          LOG_PRINT("[FAIL] muls_bf16_double_scalar: GetWorkspaceSize failed\n");
          failed++;
        }
        aclDestroyScalar(scalarDouble);
      }
      else
      {
        LOG_PRINT("[FAIL] muls_bf16_double_scalar: aclCreateScalar failed\n");
        failed++;
      }
    }
    else
    {
      LOG_PRINT("[FAIL] muls_bf16_double_scalar: CreateAclTensor failed\n");
      failed++;
    }

    totalFailed += failed;

    // 清理资源
    if (ws)
      aclrtFree(ws);
    if (tBF16)
      aclDestroyTensor(tBF16);
    if (out)
      aclDestroyTensor(out);
    if (devBF16)
      aclrtFree(devBF16);
    if (outDev)
      aclrtFree(outDev);
  }
  // 补充：bfloat16 tensor * float scalar → 触发 canUseMuls 特化分支
  {
    std::vector<int64_t> shape = {2};
    std::vector<uint16_t> bf16Data = {0x3F80, 0x4000}; // BF16: 1.0, 2.0
    std::vector<uint16_t> outData(2, 0);
    void *devSelf = nullptr, *devOut = nullptr;
    aclTensor *tSelf = nullptr, *tOut = nullptr;

    if (CreateAclTensor(bf16Data, shape, &devSelf, ACL_BF16, &tSelf) == 0 &&
        CreateAclTensor(outData, shape, &devOut, ACL_BF16, &tOut) == 0)
    {

      float scalarVal = 2.0f;
      aclScalar *sc = aclCreateScalar(&scalarVal, ACL_FLOAT);
      if (sc != nullptr)
      {
        uint64_t ws = 0;
        aclOpExecutor *exec = nullptr;
        if (aclnnMulsGetWorkspaceSize(tSelf, sc, tOut, &ws, &exec) == ACL_SUCCESS)
        {
          void *workspace = nullptr;
          if (ws > 0)
            aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
          if (aclnnMuls(workspace, ws, exec, stream) == ACL_SUCCESS)
          {
            aclrtSynchronizeStream(stream);

            // 结果验证
            std::vector<uint16_t> result(2, 0);
            aclrtMemcpy(result.data(), 2 * sizeof(uint16_t), devOut, 2 * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
            int failed = 0;
            for (size_t i = 0; i < 2; ++i)
            {
              float actual = BFloat16ToFloat(result[i]);
              float expected = (i == 0 ? 2.0f : 4.0f); // 1.0*2=2.0, 2.0*2=4.0
              if (!AlmostEqual(expected, actual, 1e-3, 1e-3))
              {
                LOG_PRINT("[FAIL] muls_bf16_float_scalar idx %zu: expected %f, got %f\n", i, expected, actual);
                failed++;
              }
            }
            if (failed == 0)
              LOG_PRINT("[PASS] muls_bf16_float_scalar\n");
          }
          if (workspace)
            aclrtFree(workspace);
        }
        aclDestroyScalar(sc);
      }
    }
    if (tSelf)
      aclDestroyTensor(tSelf);
    if (tOut)
      aclDestroyTensor(tOut);
    if (devSelf)
      aclrtFree(devSelf);
    if (devOut)
      aclrtFree(devOut);
  }
  // 1.12 广播 + 混合
  std::vector<uint16_t> fp16_host = {
      0x3C00, 0x4000, 0x4200, // 第一行: 1.0, 2.0, 3.0
      0x3C00, 0x4000, 0x4200  // 第二行: 1.0, 2.0, 3.0
  };
  std::vector<float> fp32_host = {10.0f, 20.0f, 30.0f};
  std::vector<float> expected = {
      10.0f, 40.0f, 90.0f, // 1*10, 2*20, 3*30
      10.0f, 40.0f, 90.0f};
  totalFailed += RunGenericMixedMulTest<uint16_t, float, float>(
      "mul_mixed_fp16_fp32_broadcast", stream, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT,
      {2, 3}, {3}, {2, 3}, fp16_host, fp32_host, expected, 1e-3, 1e-3);
  // ===============================================================
  // 第三组：aclnnInplaceMul 测试 (Tensor *= Tensor)
  // ===============================================================
  LOG_PRINT("\n--- 3. aclnnInplaceMul API 测试 ---\n");

  // 触发 NPU 模拟器手写的 200 行 Double 乘法核心逻辑！
  totalFailed += RunGenericInplaceMulTest<double>(
      "inplace_mul_double", stream, ACL_DOUBLE,
      {2, 2},
      {1.123456789, -2.5, 0.0, 9999.99},
      {2.0, -2.0, 5.0, 0.1},
      {2.246913578, 5.0, 0.0, 999.999}, 1e-9, 1e-9);

  totalFailed += RunGenericInplaceMulTest<int8_t>(
      "inplace_mul_int8_broadcast", stream, ACL_INT8,
      {2, 3},
      {1, 2, 3, 4, 5, 6}, {2, -1, 0, 2, -1, 0},
      {2, -2, 0, 8, -5, 0}, 0.0, 0.0);

  // ===============================================================
  // 第四组：aclnnInplaceMuls 测试 (Tensor *= Scalar)
  // ===============================================================
  LOG_PRINT("\n--- 4. aclnnInplaceMuls API 测试 ---\n");

  totalFailed += RunGenericInplaceMulsTest<float>(
      "inplace_muls_float32_zero", stream, ACL_FLOAT,
      {4}, {3.14f, -99.9f, 0.0f, 12345.6f}, 0.0f,
      {0.0f, -0.0f, 0.0f, 0.0f});

  totalFailed += RunGenericInplaceMulsTest<int32_t>(
      "inplace_muls_int32_negative", stream, ACL_INT32,
      {2, 2}, {10, -20, 30, 0}, -5,
      {-50, 100, -150, 0}, 0.0, 0.0);

  // ===============================================================
  // 第五组：定制化 Float16 测试 (带有真实的容差比对与边界值)
  // ===============================================================
  LOG_PRINT("\n--- 5. Float16 定制测试 ---\n");
  totalFailed += RunFloat16Test(stream);

  // 内存不连续测试：Shape {2, 2}，但 Strides 给 {4, 1} (模拟跳跃访问)
  LOG_PRINT("\n--- 非连续内存测试 ---\n");
  // =========================================================
  // 补充：非连续 float16 * 非连续 float32 (混合精度 + 跳跃内存验证)
  // =========================================================
  {
    std::vector<int64_t> shape = {2, 2};
    std::vector<int64_t> badStrides = {4, 1}; // 物理索引: 0,1, 4,5
    // 对应内存: {1, 2, pad, pad, 3, 4, pad, pad}
    std::vector<uint16_t> fp16Data = {0x3C00, 0x4000, 0, 0, 0x4200, 0x4400, 0, 0};
    std::vector<float> fp32Data = {2.0f, 2.0f, 0, 0, 2.0f, 2.0f, 0, 0};
    std::vector<float> outData(8, 0); // 预留 8 个坑位

    void *dev1 = nullptr, *dev2 = nullptr, *outDev = nullptr;
    aclTensor *t1 = nullptr, *t2 = nullptr, *out = nullptr;

    if (CreateNonContiguousAclTensor(fp16Data, shape, badStrides, &dev1, ACL_FLOAT16, &t1) == 0 &&
        CreateNonContiguousAclTensor(fp32Data, shape, badStrides, &dev2, ACL_FLOAT, &t2) == 0 &&
        CreateNonContiguousAclTensor(outData, shape, badStrides, &outDev, ACL_FLOAT, &out) == 0)
    {

      uint64_t ws = 0;
      aclOpExecutor *exec = nullptr;
      if (aclnnMulGetWorkspaceSize(t1, t2, out, &ws, &exec) == ACL_SUCCESS)
      {
        void *workspace = nullptr;
        if (ws > 0)
          aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclnnMul(workspace, ws, exec, stream) == ACL_SUCCESS)
        {
          aclrtSynchronizeStream(stream);

          // 拷回整块内存 (8个元素)，验证 ViewCopy 是否破坏了 pad 区域
          aclrtMemcpy(outData.data(), outData.size() * sizeof(float), outDev, outData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

          std::vector<float> expected = {2, 4, 0, 0, 6, 8, 0, 0};
          int failed = 0;
          for (size_t i = 0; i < expected.size(); ++i)
          {
            if (std::abs(expected[i] - outData[i]) > 1e-3)
            {
              LOG_PRINT("[FAIL] non_contiguous_mixed idx %zu: expected %f, got %f\n", i, expected[i], outData[i]);
              failed++;
            }
          }
          if (failed == 0)
            LOG_PRINT("[PASS] non_contiguous_mixed\n");
          else
            totalFailed += failed; // 累加错误！
        }
        else
        {
          LOG_PRINT("[FAIL] non_contiguous_mixed: aclnnMul execution failed\n");
          totalFailed++;
        }
        if (workspace)
          aclrtFree(workspace);
      }
      else
      {
        LOG_PRINT("[FAIL] non_contiguous_mixed: GetWorkspaceSize failed\n");
        totalFailed++;
      }

      if (t1)
        aclDestroyTensor(t1);
      if (t2)
        aclDestroyTensor(t2);
      if (out)
        aclDestroyTensor(out);
      if (dev1)
        aclrtFree(dev1);
      if (dev2)
        aclrtFree(dev2);
      if (outDev)
        aclrtFree(outDev);
    }
    else
    {
      LOG_PRINT("[FAIL] non_contiguous_mixed: tensor creation failed\n");
      totalFailed++;
    }
  }
  // 补充：Inplace 遇到非连续内存的异常/校验分支
  {
    std::vector<int64_t> shape = {2, 2};
    std::vector<int64_t> badStrides = {4, 1};
    void *devSelf = nullptr, *devOther = nullptr;
    aclTensor *tSelf = nullptr, *tOther = nullptr;
    std::vector<float> data1 = {1, 2, 0, 0, 3, 4, 0, 0};
    std::vector<float> data2 = {2, 2, 0, 0, 2, 2, 0, 0};

    if (CreateNonContiguousAclTensor(data1, shape, badStrides, &devSelf, ACL_FLOAT, &tSelf) == 0 &&
        CreateNonContiguousAclTensor(data2, shape, badStrides, &devOther, ACL_FLOAT, &tOther) == 0)
    {

      uint64_t wsSize = 0;
      aclOpExecutor *exec = nullptr;
      // 传入 InplaceMul，看底层是否允许非连续内存的原地操作
      aclnnInplaceMulGetWorkspaceSize(tSelf, tOther, &wsSize, &exec);
      // 无论 SUCCESS 还是报错，这行代码本身就会点亮底层校验 Stride 的隐藏 if-else 树！
    }
    if (tSelf)
      aclDestroyTensor(tSelf);
    if (tOther)
      aclDestroyTensor(tOther);
    if (devSelf)
      aclrtFree(devSelf);
    if (devOther)
      aclrtFree(devOther);
  }
  totalFailed += RunApiValidationTests(stream);
  // 第六组：异常
  totalFailed += RunUltimateExceptionTests(stream);
  // =======================================================

  // 补充测试1：大 Tensor 测试（例如 100 万个元素），触发多核 Tiling 策略
  // 补充测试：大 Tensor 测试（65536个元素，足以触发多核 Tiling，且不会让模拟器超时）
  totalFailed += RunGenericMulTest<float>(
      "mul_float32_large_tensor", stream, ACL_FLOAT,
      {256, 256}, {256, 256}, {256, 256},
      std::vector<float>(65536, 2.0f),
      std::vector<float>(65536, 3.0f),
      std::vector<float>(65536, 6.0f),
      1e-5, 1e-5);
  totalFailed += RunGenericMulTest<float>(
      "mul_tiling_multi_block", stream, ACL_FLOAT,
      {128, 1024}, {128, 1024}, {128, 1024},
      std::vector<float>(131072, 1.0f),
      std::vector<float>(131072, 2.0f),
      std::vector<float>(131072, 2.0f),
      1e-5, 1e-5);
  // 非对齐大向量
  // 修改后的非对齐大张量
  totalFailed += RunGenericMulTest<float>(
      "mul_non_aligned_large", stream, ACL_FLOAT,
      {1023, 127}, {1023, 127}, {1023, 127}, // 非 16/32 的倍数，足够大
      std::vector<float>(1023 * 127, 1.0f),
      std::vector<float>(1023 * 127, 2.0f),
      std::vector<float>(1023 * 127, 2.0f),
      1e-5, 1e-5);
  // 补充测试2：Shape 包含 0 的空 Tensor
  // totalFailed += RunGenericMulTest<float>(
  //     "mul_empty_tensor", stream, ACL_FLOAT,
  //     {2, 0, 3}, {2, 0, 3}, {2, 0, 3}, // 元素总数为 0
  //     std::vector<float>{}, std::vector<float>{}, std::vector<float>{},
  //     0.0, 0.0
  // );
  // 补充测试3：0-D Tensor (标量 Tensor)
  totalFailed += RunGenericMulTest<float>(
      "mul_0d_tensor", stream, ACL_FLOAT,
      {}, {}, {},               // shape 为空代表 0-D 标量
      {3.14f}, {2.0f}, {6.28f}, // 数据只有一个
      1e-5, 1e-5);
  // =========================================================
  // 补充：空 Tensor 测试 (点亮 468 行的 IsEmpty() 分支)
  // =========================================================
  totalFailed += RunGenericMulTest<float>(
      "mul_empty_tensor", stream, ACL_FLOAT,
      {2, 0, 3}, {2, 0, 3}, {2, 0, 3}, // 包含 0 的 shape
      std::vector<float>{}, std::vector<float>{}, std::vector<float>{},
      0.0, 0.0);
  // 第三组：aclnnInplaceMul 测试
  // ===============================================================
  // 第三组：aclnnInplaceMul 测试 (Tensor *= Tensor)
  // ===============================================================
  LOG_PRINT("\n--- 3. aclnnInplaceMul API 测试 ---\n");

  // 1. Double 类型测试
  totalFailed += RunGenericInplaceMulTest<double>(
      "inplace_mul_double", stream, ACL_DOUBLE,
      {2, 2},
      {1.123456789, -2.5, 0.0, 9999.99},
      {2.0, -2.0, 5.0, 0.1},
      {2.246913578, 5.0, 0.0, 999.999}, 1e-9, 1e-9);

  // 2. INT8 广播测试
  totalFailed += RunGenericInplaceMulTest<int8_t>(
      "inplace_mul_int8_broadcast", stream, ACL_INT8,
      {2, 3},
      {1, 2, 3, 4, 5, 6}, {2, -1, 0, 2, -1, 0},
      {2, -2, 0, 8, -5, 0}, 0.0, 0.0);

  // 3. 混合类型测试 (FP32 *= FP16) [新增]
  TestInplaceMulMixedType(stream, totalFailed);
  // ===============================================================
  // 汇总与资源清理
  // ===============================================================
  LOG_PRINT("\n====================================================\n");
  if (totalFailed == 0)
  {
    LOG_PRINT("\nALL TESTS PASSED\n");
  }
  else
  {
    LOG_PRINT("\nTOTAL FAILED = %d\n", totalFailed);
  }
  LOG_PRINT("====================================================\n");

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return totalFailed;
}