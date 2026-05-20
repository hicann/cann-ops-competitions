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
#include <string>
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

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

// ==================== 测试结果验证框架 ====================

// 测试统计
struct TestStats {
  int passed = 0;
  int failed = 0;
  
  void record(bool pass) {
    if (pass) passed++; else failed++;
  }
  
  void printSummary() const {
    printf("\n========== Test Summary ==========\n");
    printf("Total: %d, Passed: %d, Failed: %d\n", passed + failed, passed, failed);
    if (failed == 0) {
      printf("All tests PASSED!\n");
    } else {
      printf("Some tests FAILED!\n");
    }
    printf("==================================\n");
  }
};

// 容差比较函数：|actual - expected| <= atol + rtol * |expected|
bool CompareResult(const float* actual, const float* expected, int64_t size,
                   float atol, float rtol) {
  for (int64_t i = 0; i < size; i++) {
    float diff = std::fabs(actual[i] - expected[i]);
    float tolerance = atol + rtol * std::fabs(expected[i]);
    if (diff > tolerance) {
      printf("  Mismatch at index %ld: actual=%f, expected=%f, diff=%f, tolerance=%f\n",
             i, actual[i], expected[i], diff, tolerance);
      return false;
    }
  }
  return true;
}

// 前向声明
int64_t GetShapeSize(const std::vector<int64_t>& shape);

// CPU参考计算：逐元素乘法
template <typename T>
void ReferenceMul(const T* self, const T* other, T* out, int64_t size) {
  for (int64_t i = 0; i < size; i++) {
    out[i] = self[i] * other[i];
  }
}

// CPU参考计算：广播支持的逐元素乘法
template <typename T>
void ReferenceMulBroadcast(
    const T* self, const std::vector<int64_t>& selfShape,
    const T* other, const std::vector<int64_t>& otherShape,
    T* out, const std::vector<int64_t>& outShape) {
  
  int64_t outSize = GetShapeSize(outShape);
  int64_t selfDim = selfShape.size();
  int64_t otherDim = otherShape.size();
  int64_t outDim = outShape.size();
  
  for (int64_t i = 0; i < outSize; i++) {
    int64_t tmp = i;
    int64_t selfIdx = 0;
    int64_t otherIdx = 0;
    int64_t selfStride = 1;
    int64_t otherStride = 1;
    
    // 从后向前计算各维度索引
    for (int64_t d = outDim - 1; d >= 0; d--) {
      int64_t coord = tmp % outShape[d];
      tmp /= outShape[d];
      
      // 计算self索引（考虑广播）
      if (outDim - 1 - d < selfDim) {
        int64_t selfD = selfShape[selfDim - 1 - (outDim - 1 - d)];
        if (selfD > 1) {
          selfIdx += coord * selfStride;
        }
        selfStride *= selfD;
      }
      
      // 计算other索引（考虑广播）
      if (outDim - 1 - d < otherDim) {
        int64_t otherD = otherShape[otherDim - 1 - (outDim - 1 - d)];
        if (otherD > 1) {
          otherIdx += coord * otherStride;
        }
        otherStride *= otherD;
      }
    }
    
    out[i] = self[selfIdx] * other[otherIdx];
  }
}

// 计算广播后的输出shape
std::vector<int64_t> BroadcastShape(
    const std::vector<int64_t>& shape1,
    const std::vector<int64_t>& shape2) {
  int64_t ndim1 = shape1.size();
  int64_t ndim2 = shape2.size();
  int64_t outNdim = (ndim1 > ndim2) ? ndim1 : ndim2;
  
  std::vector<int64_t> outShape(outNdim);
  
  for (int64_t i = 0; i < outNdim; i++) {
    int64_t dim1 = (i < ndim1) ? shape1[ndim1 - 1 - i] : 1;
    int64_t dim2 = (i < ndim2) ? shape2[ndim2 - 1 - i] : 1;
    
    if (dim1 == 1) {
      outShape[outNdim - 1 - i] = dim2;
    } else if (dim2 == 1) {
      outShape[outNdim - 1 - i] = dim1;
    } else if (dim1 == dim2) {
      outShape[outNdim - 1 - i] = dim1;
    } else {
      // 广播不兼容，返回空shape表示错误
      return {};
    }
  }
  
  return outShape;
}

// 打印测试结果
void PrintTestResult(const std::string& testName, bool pass) {
  if (pass) {
    printf("[PASS] %s\n", testName.c_str());
  } else {
    printf("[FAIL] %s\n", testName.c_str());
  }
}

// 容差配置（题目要求）
struct ToleranceConfig {
  static constexpr float FLOAT32_ATOL = 1e-5f;
  static constexpr float FLOAT32_RTL = 1e-5f;
  static constexpr float FLOAT16_ATOL = 1e-3f;
  static constexpr float FLOAT16_RTL = 1e-3f;
  static constexpr float BF16_ATOL = 1e-2f;
  static constexpr float BF16_RTL = 1e-2f;
};

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
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
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

// ==================== 测试用例实现 ====================

// ==================== 数据类型转换辅助函数 ====================

// float32 -> float16 (IEEE 754 half precision)
uint16_t Float32ToFloat16(float val) {
  uint32_t f = *reinterpret_cast<uint32_t*>(&val);
  uint32_t sign = (f >> 31) & 0x1;
  uint32_t exp = (f >> 23) & 0xFF;
  uint32_t mant = f & 0x7FFFFF;
  
  uint16_t h;
  if (exp == 0) {
    h = (sign << 15) | 0; // zero or denorm
  } else if (exp == 255) {
    h = (sign << 15) | 0x7C00 | (mant ? 0x200 : 0); // inf or nan
  } else {
    int32_t newExp = (int32_t)exp - 127 + 15;
    if (newExp >= 31) {
      h = (sign << 15) | 0x7C00; // inf
    } else if (newExp <= 0) {
      h = (sign << 15) | 0; // zero or denorm
    } else {
      h = (sign << 15) | (newExp << 10) | (mant >> 13);
    }
  }
  return h;
}

// float16 -> float32
float Float16ToFloat32(uint16_t val) {
  uint32_t sign = (val >> 15) & 0x1;
  uint32_t exp = (val >> 10) & 0x1F;
  uint32_t mant = val & 0x3FF;
  
  uint32_t f;
  if (exp == 0) {
    f = (sign << 31); // zero or denorm
  } else if (exp == 31) {
    f = (sign << 31) | 0x7F800000 | (mant ? 0x400000 : 0); // inf or nan
  } else {
    f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  }
  return *reinterpret_cast<float*>(&f);
}

// ==================== 测试用例实现 ====================

// Step 1: 基础测试 - FLOAT32同shape测试
bool TestMulFloat32Basic(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Mul_FLOAT32_Basic [4,2]*[4,2]\n");
  bool pass = false;
  void* workspaceAddr = nullptr;
  
  // 构造输入与输出
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(8, 0);
  
  // 创建tensor
  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Mul_FLOAT32_Basic", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Mul_FLOAT32_Basic", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Mul_FLOAT32_Basic", false);
    stats.record(false);
    return false;
  }

  // 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  
  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      printf("  allocate workspace failed: %d\n", ret);
    }
  }
  
  if (ret == ACL_SUCCESS) {
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
      printf("  aclnnMul failed: %d\n", ret);
    }
  } else {
    printf("  aclnnMulGetWorkspaceSize failed: %d\n", ret);
  }

  // 同步等待
  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
      printf("  aclrtSynchronizeStream failed: %d\n", ret);
    }
  }

  // 获取输出结果并验证
  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
      printf("  copy result failed: %d\n", ret);
    } else {
      // CPU参考计算
      std::vector<float> expectedData(size, 0);
      ReferenceMul(selfHostData.data(), otherHostData.data(), expectedData.data(), size);
      
      // 打印期望与实际值（调试用）
      printf("  Expected: ");
      for (int64_t i = 0; i < size; i++) printf("%f ", expectedData[i]);
      printf("\n");
      printf("  Actual:   ");
      for (int64_t i = 0; i < size; i++) printf("%f ", resultData[i]);
      printf("\n");

      // 结果验证
      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  // 释放workspace
  if (workspaceSize > 0 && workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  // 释放资源
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  
  PrintTestResult("Mul_FLOAT32_Basic", pass);
  stats.record(pass);
  return pass;
}

// Step 2: FLOAT16测试
bool TestMulFloat16Basic(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Mul_FLOAT16_Basic [4,2]*[4,2]\n");
  bool pass = false;
  void* workspaceAddr = nullptr;
  
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  
  // FLOAT16数据（用uint16_t存储）
  std::vector<float> selfFloatData = {0.5f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<float> otherFloatData = {2.0f, 1.0f, 0.5f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f};
  
  std::vector<uint16_t> selfHostData(8);
  std::vector<uint16_t> otherHostData(8);
  std::vector<uint16_t> outHostData(8, 0);
  
  for (int i = 0; i < 8; i++) {
    selfHostData[i] = Float32ToFloat16(selfFloatData[i]);
    otherHostData[i] = Float32ToFloat16(otherFloatData[i]);
  }
  
  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Mul_FLOAT16_Basic", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT16, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Mul_FLOAT16_Basic", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Mul_FLOAT16_Basic", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  
  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }
  
  if (ret == ACL_SUCCESS) {
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(uint16_t), outDeviceAddr,
                      size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
      // CPU参考计算（用float计算再转回）
      std::vector<float> expectedFloat(size, 0);
      ReferenceMul(selfFloatData.data(), otherFloatData.data(), expectedFloat.data(), size);
      
      // 转换结果到float比较
      std::vector<float> resultFloat(size);
      for (int64_t i = 0; i < size; i++) {
        resultFloat[i] = Float16ToFloat32(resultData[i]);
      }
      
      printf("  Expected: ");
      for (int64_t i = 0; i < size; i++) printf("%f ", expectedFloat[i]);
      printf("\n");
      printf("  Actual:   ");
      for (int64_t i = 0; i < size; i++) printf("%f ", resultFloat[i]);
      printf("\n");

      pass = CompareResult(resultFloat.data(), expectedFloat.data(), size,
                           ToleranceConfig::FLOAT16_ATOL, ToleranceConfig::FLOAT16_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  
  PrintTestResult("Mul_FLOAT16_Basic", pass);
  stats.record(pass);
  return pass;
}

// Step 2: 混合类型测试 - FLOAT16 * FLOAT -> FLOAT
bool TestMulMixedFp16Fp32(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Mul_MIXED_FP16_FP32 [4,2]*[4,2]\n");
  bool pass = false;
  void* workspaceAddr = nullptr;
  
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;
  
  // self: FP16, other: FP32, out: FP32
  std::vector<float> selfFloatData = {0.5f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<float> otherHostData = {2.0f, 1.0f, 0.5f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f};
  
  std::vector<uint16_t> selfHostData(8);
  for (int i = 0; i < 8; i++) {
    selfHostData[i] = Float32ToFloat16(selfFloatData[i]);
  }
  std::vector<float> outHostData(8, 0);
  
  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Mul_MIXED_FP16_FP32", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Mul_MIXED_FP16_FP32", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Mul_MIXED_FP16_FP32", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  
  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }
  
  if (ret == ACL_SUCCESS) {
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
      // CPU参考计算（全部用float）
      std::vector<float> expectedData(size, 0);
      ReferenceMul(selfFloatData.data(), otherHostData.data(), expectedData.data(), size);
      
      printf("  Expected: ");
      for (int64_t i = 0; i < size; i++) printf("%f ", expectedData[i]);
      printf("\n");
      printf("  Actual:   ");
      for (int64_t i = 0; i < size; i++) printf("%f ", resultData[i]);
      printf("\n");

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("Mul_MIXED_FP16_FP32", pass);
  stats.record(pass);
  return pass;
}

// Step 3: 广播测试 - 行广播 [4,3] * [3]
bool TestMulBroadcastRow(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Mul_Broadcast_Row [4,3]*[3]\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> selfShape = {4, 3};
  std::vector<int64_t> otherShape = {3};
  std::vector<int64_t> outShape = {4, 3};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  // 输入数据
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};  // [4,3]
  std::vector<float> otherHostData = {2, 3, 4};  // [3]，将广播到 [4,3]
  std::vector<float> outHostData(12, 0);

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Mul_Broadcast_Row", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Mul_Broadcast_Row", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Mul_Broadcast_Row", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
      // CPU参考计算（带广播）
      std::vector<float> expectedData(size, 0);
      ReferenceMulBroadcast(
        selfHostData.data(), selfShape,
        otherHostData.data(), otherShape,
        expectedData.data(), outShape
      );

      printf("  Expected: ");
      for (int64_t i = 0; i < size; i++) printf("%f ", expectedData[i]);
      printf("\n");
      printf("  Actual:   ");
      for (int64_t i = 0; i < size; i++) printf("%f ", resultData[i]);
      printf("\n");

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("Mul_Broadcast_Row", pass);
  stats.record(pass);
  return pass;
}

// Step 3: 广播测试 - 列广播 [4,3] * [4,1]
bool TestMulBroadcastCol(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Mul_Broadcast_Col [4,3]*[4,1]\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> selfShape = {4, 3};
  std::vector<int64_t> otherShape = {4, 1};
  std::vector<int64_t> outShape = {4, 3};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  // 输入数据
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};  // [4,3]
  std::vector<float> otherHostData = {2, 3, 4, 5};  // [4,1]，将广播到 [4,3]
  std::vector<float> outHostData(12, 0);

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Mul_Broadcast_Col", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Mul_Broadcast_Col", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Mul_Broadcast_Col", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
      // CPU参考计算（带广播）
      std::vector<float> expectedData(size, 0);
      ReferenceMulBroadcast(
        selfHostData.data(), selfShape,
        otherHostData.data(), otherShape,
        expectedData.data(), outShape
      );

      printf("  Expected: ");
      for (int64_t i = 0; i < size; i++) printf("%f ", expectedData[i]);
      printf("\n");
      printf("  Actual:   ");
      for (int64_t i = 0; i < size; i++) printf("%f ", resultData[i]);
      printf("\n");

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("Mul_Broadcast_Col", pass);
  stats.record(pass);
  return pass;
}

// Step 3: 广播测试 - 多维度广播 [2,1,4] * [1,3,1]
bool TestMulBroadcastMultiDim(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Mul_Broadcast_MultiDim [2,1,4]*[1,3,1]\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> selfShape = {2, 1, 4};
  std::vector<int64_t> otherShape = {1, 3, 1};
  std::vector<int64_t> outShape = {2, 3, 4};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  // 输入数据
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};  // [2,1,4]
  std::vector<float> otherHostData = {1, 2, 3};  // [1,3,1]，将广播到 [2,3,4]
  std::vector<float> outHostData(24, 0);

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Mul_Broadcast_MultiDim", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Mul_Broadcast_MultiDim", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Mul_Broadcast_MultiDim", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
      // CPU参考计算（带广播）
      std::vector<float> expectedData(size, 0);
      ReferenceMulBroadcast(
        selfHostData.data(), selfShape,
        otherHostData.data(), otherShape,
        expectedData.data(), outShape
      );

      printf("  Expected: ");
      for (int64_t i = 0; i < size; i++) printf("%f ", expectedData[i]);
      printf("\n");
      printf("  Actual:   ");
      for (int64_t i = 0; i < size; i++) printf("%f ", resultData[i]);
      printf("\n");

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("Mul_Broadcast_MultiDim", pass);
  stats.record(pass);
  return pass;
}

// Step 4: 测试4个API变体 - Mul/Muls/InplaceMul/InplaceMuls

// 4.1 测试aclnnMuls - 张量乘以标量
bool TestMulsFloat32(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Muls_FLOAT32 [4,2]*scalar(2.0)\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  // 准备数据
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<float> selfHostData = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float scalarValue = 2.0f;
  std::vector<int64_t> outShape = {4, 2};
  std::vector<float> outHostData(8, 0);
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  // 创建aclTensor
  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Muls_FLOAT32", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Muls_FLOAT32", false);
    stats.record(false);
    return false;
  }

  // 创建aclScalar
  aclScalar* other = aclCreateScalar(&scalarValue, ACL_FLOAT);
  if (other == nullptr) {
    printf("  create scalar failed\n");
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    PrintTestResult("Muls_FLOAT32", false);
    stats.record(false);
    return false;
  }

  // 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulsGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
      // CPU参考计算：每个元素乘以标量
      std::vector<float> expectedData(size, 0);
      for (int64_t i = 0; i < size; i++) {
        expectedData[i] = selfHostData[i] * scalarValue;
      }

      printf("  Scalar: %f\n", scalarValue);
      printf("  Expected: ");
      for (int64_t i = 0; i < size; i++) printf("%f ", expectedData[i]);
      printf("\n");
      printf("  Actual:   ");
      for (int64_t i = 0; i < size; i++) printf("%f ", resultData[i]);
      printf("\n");

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyScalar(other);
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("Muls_FLOAT32", pass);
  stats.record(pass);
  return pass;
}

// 4.2 测试aclnnInplaceMul - 原地乘法 (self = self * other)
bool TestInplaceMulFloat32(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: InplaceMul_FLOAT32 [4,2]*=[4,2]\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  // 准备数据
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<float> selfHostData = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<float> selfOriginal = selfHostData; // 保存原始值用于验证
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<float> otherHostData = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;

  // 创建aclTensor（self会作为输入和输出）
  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("InplaceMul_FLOAT32", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("InplaceMul_FLOAT32", false);
    stats.record(false);
    return false;
  }

  // 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceMulGetWorkspaceSize(self, other, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(selfShape);
    std::vector<float> resultData(size, 0);
    // 从self的device地址拷贝结果（原地修改）
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), selfDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
      // CPU参考计算
      std::vector<float> expectedData(size, 0);
      ReferenceMul(selfOriginal.data(), otherHostData.data(), expectedData.data(), size);

      printf("  Original: ");
      for (int64_t i = 0; i < size; i++) printf("%f ", selfOriginal[i]);
      printf("\n");
      printf("  Other:    ");
      for (int64_t i = 0; i < size; i++) printf("%f ", otherHostData[i]);
      printf("\n");
      printf("  Expected: ");
      for (int64_t i = 0; i < size; i++) printf("%f ", expectedData[i]);
      printf("\n");
      printf("  Actual:   ");
      for (int64_t i = 0; i < size; i++) printf("%f ", resultData[i]);
      printf("\n");

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);

  PrintTestResult("InplaceMul_FLOAT32", pass);
  stats.record(pass);
  return pass;
}

// 4.3 测试aclnnInplaceMuls - 原地标量乘法 (self = self * scalar)
bool TestInplaceMulsFloat32(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: InplaceMuls_FLOAT32 [4,2]*=scalar(3.0)\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  // 准备数据
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> selfOriginal = selfHostData; // 保存原始值用于验证
  float scalarValue = 3.0f;
  void* selfDeviceAddr = nullptr;
  aclTensor* self = nullptr;

  // 创建aclTensor（self会作为输入和输出）
  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("InplaceMuls_FLOAT32", false);
    stats.record(false);
    return false;
  }

  // 创建aclScalar
  aclScalar* other = aclCreateScalar(&scalarValue, ACL_FLOAT);
  if (other == nullptr) {
    printf("  create scalar failed\n");
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("InplaceMuls_FLOAT32", false);
    stats.record(false);
    return false;
  }

  // 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnInplaceMulsGetWorkspaceSize(self, other, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(selfShape);
    std::vector<float> resultData(size, 0);
    // 从self的device地址拷贝结果（原地修改）
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), selfDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
      // CPU参考计算：每个元素乘以标量
      std::vector<float> expectedData(size, 0);
      for (int64_t i = 0; i < size; i++) {
        expectedData[i] = selfOriginal[i] * scalarValue;
      }

      printf("  Scalar: %f\n", scalarValue);
      printf("  Expected: ");
      for (int64_t i = 0; i < size; i++) printf("%f ", expectedData[i]);
      printf("\n");
      printf("  Actual:   ");
      for (int64_t i = 0; i < size; i++) printf("%f ", resultData[i]);
      printf("\n");

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  aclDestroyTensor(self);
  aclDestroyScalar(other);
  aclrtFree(selfDeviceAddr);

  PrintTestResult("InplaceMuls_FLOAT32", pass);
  stats.record(pass);
  return pass;
}

// Step 5: 边界值和异常测试

// 5.1 边界值测试 - 大数值
bool TestMulBoundaryLarge(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Mul_Boundary_Large [2,2]*[2,2] (large values)\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {1e10f, 1e15f, 1e20f, 1e25f};
  std::vector<float> otherHostData = {2.0f, 3.0f, 0.5f, 0.1f};
  std::vector<float> outHostData(4, 0);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  auto ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Mul_Boundary_Large", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Mul_Boundary_Large", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Mul_Boundary_Large", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
      std::vector<float> expectedData(size, 0);
      ReferenceMul(selfHostData.data(), otherHostData.data(), expectedData.data(), size);

      printf("  Testing large values: 1e10, 1e15, 1e20, 1e25\n");
      printf("  Expected: ");
      for (int64_t i = 0; i < size; i++) printf("%.2e ", expectedData[i]);
      printf("\n");
      printf("  Actual:   ");
      for (int64_t i = 0; i < size; i++) printf("%.2e ", resultData[i]);
      printf("\n");

      pass = CompareResult(resultData.data(), expectedData.data(), size, 1e-3f, 1e-4f);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("Mul_Boundary_Large", pass);
  stats.record(pass);
  return pass;
}

// 5.2 边界值测试 - 小数值和零值
bool TestMulBoundarySmallAndZero(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Mul_Boundary_Small_Zero [2,3]*[2,3] (small values + zero)\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHostData = {0.0f, 1e-10f, 1e-5f, 0.0f, 1e-8f, 1e-3f};
  std::vector<float> otherHostData = {1e-5f, 1e5f, 2.0f, 0.0f, 1e8f, 0.5f};
  std::vector<float> outHostData(6, 0);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  auto ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Mul_Boundary_Small_Zero", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Mul_Boundary_Small_Zero", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Mul_Boundary_Small_Zero", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
      std::vector<float> expectedData(size, 0);
      ReferenceMul(selfHostData.data(), otherHostData.data(), expectedData.data(), size);

      printf("  Testing small values and zero\n");
      printf("  Expected: ");
      for (int64_t i = 0; i < size; i++) printf("%.2e ", expectedData[i]);
      printf("\n");
      printf("  Actual:   ");
      for (int64_t i = 0; i < size; i++) printf("%.2e ", resultData[i]);
      printf("\n");

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("Mul_Boundary_Small_Zero", pass);
  stats.record(pass);
  return pass;
}

// 5.3 边界值测试 - 负数测试
bool TestMulBoundaryNegative(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Mul_Boundary_Negative [2,2]*[2,2] (negative values)\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfHostData = {-1.0f, 2.0f, -3.0f, 4.0f};
  std::vector<float> otherHostData = {2.0f, -3.0f, -4.0f, 5.0f};
  std::vector<float> outHostData(4, 0);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  auto ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Mul_Boundary_Negative", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Mul_Boundary_Negative", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Mul_Boundary_Negative", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

  if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  }

  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }

  if (ret == ACL_SUCCESS) {
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
      std::vector<float> expectedData(size, 0);
      ReferenceMul(selfHostData.data(), otherHostData.data(), expectedData.data(), size);

      printf("  Testing negative values\n");
      printf("  Expected: ");
      for (int64_t i = 0; i < size; i++) printf("%.1f ", expectedData[i]);
      printf("\n");
      printf("  Actual:   ");
      for (int64_t i = 0; i < size; i++) printf("%.1f ", resultData[i]);
      printf("\n");

      pass = CompareResult(resultData.data(), expectedData.data(), size,
                           ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
    }
  }

  if (workspaceSize > 0 && workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("Mul_Boundary_Negative", pass);
  stats.record(pass);
  return pass;
}

// 5.4 异常测试 - 不兼容的shape（应该返回错误）
bool TestMulIncompatibleShape(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Mul_Incompatible_Shape [3,2]*[4,2] (should fail)\n");
  bool pass = false;

  std::vector<int64_t> selfShape = {3, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  std::vector<float> selfHostData(6, 1.0f);
  std::vector<float> otherHostData(8, 2.0f);
  std::vector<float> outHostData(8, 0);
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    PrintTestResult("Mul_Incompatible_Shape", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclrtFree(selfDeviceAddr);
    PrintTestResult("Mul_Incompatible_Shape", false);
    stats.record(false);
    return false;
  }
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(selfDeviceAddr);
    aclrtFree(otherDeviceAddr);
    PrintTestResult("Mul_Incompatible_Shape", false);
    stats.record(false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

  if (ret != ACL_SUCCESS) {
    printf("  Expected failure: aclnnMulGetWorkspaceSize returned %d\n", ret);
    pass = true;
  } else {
    printf("  Warning: Expected failure but got success\n");
    pass = false;
  }

  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(out);
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);

  PrintTestResult("Mul_Incompatible_Shape", pass);
  stats.record(pass);
  return pass;
}

// 6.1 INT32数据类型测试
bool TestMulInt32(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Mul_INT32_Basic [4,2]*[4,2]\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> shape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  std::vector<int32_t> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32_t> otherHostData = {2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int32_t> outHostData(8, 0);

  auto size = GetShapeSize(shape);
  auto ret = aclrtMalloc(&selfDeviceAddr, size * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret == ACL_SUCCESS) {
    ret = aclrtMemcpy(selfDeviceAddr, size * sizeof(int32_t), selfHostData.data(),
                      size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
  }
  if (ret == ACL_SUCCESS) {
    ret = aclrtMalloc(&otherDeviceAddr, size * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
  }
  if (ret == ACL_SUCCESS) {
    ret = aclrtMemcpy(otherDeviceAddr, size * sizeof(int32_t), otherHostData.data(),
                      size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
  }
  if (ret == ACL_SUCCESS) {
    ret = aclrtMalloc(&outDeviceAddr, size * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
  }
  if (ret == ACL_SUCCESS) {
    ret = aclrtMemcpy(outDeviceAddr, size * sizeof(int32_t), outHostData.data(),
                      size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
  }

  if (ret == ACL_SUCCESS) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
    self = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT32, strides.data(), 0,
                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), selfDeviceAddr);
    other = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT32, strides.data(), 0,
                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), otherDeviceAddr);
    out = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT32, strides.data(), 0,
                          aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), outDeviceAddr);
  }

  if (self != nullptr && other != nullptr && out != nullptr) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

    if (ret == ACL_SUCCESS && workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    if (ret == ACL_SUCCESS) {
      ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    }

    if (ret == ACL_SUCCESS) {
      ret = aclrtSynchronizeStream(stream);
    }

    if (ret == ACL_SUCCESS) {
      std::vector<int32_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int32_t), outDeviceAddr,
                        size * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
      if (ret == ACL_SUCCESS) {
        bool allMatch = true;
        for (int64_t i = 0; i < size; i++) {
          int32_t expected = selfHostData[i] * otherHostData[i];
          if (resultData[i] != expected) {
            printf("  Mismatch at %ld: actual=%d, expected=%d\n", i, resultData[i], expected);
            allMatch = false;
            break;
          }
        }
        pass = allMatch;
        printf("  Expected: ");
        for (int64_t i = 0; i < size; i++) printf("%d ", selfHostData[i] * otherHostData[i]);
        printf("\n");
        printf("  Actual:   ");
        for (int64_t i = 0; i < size; i++) printf("%d ", resultData[i]);
        printf("\n");
      }
    }
  }

  if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  if (self != nullptr) aclDestroyTensor(self);
  if (other != nullptr) aclDestroyTensor(other);
  if (out != nullptr) aclDestroyTensor(out);
  if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
  if (otherDeviceAddr != nullptr) aclrtFree(otherDeviceAddr);
  if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);

  PrintTestResult("Mul_INT32_Basic", pass);
  stats.record(pass);
  return pass;
}

// 6.2 1D张量测试
bool TestMul1D(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Mul_1D [8]*[8]\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> shape = {8};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> otherHostData = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  std::vector<float> outHostData(8, 0.0f);

  auto ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret == ACL_SUCCESS) {
    ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  }
  if (ret == ACL_SUCCESS) {
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  }

  if (ret == ACL_SUCCESS) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

    if (ret == ACL_SUCCESS && workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    if (ret == ACL_SUCCESS) {
      ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    }

    if (ret == ACL_SUCCESS) {
      ret = aclrtSynchronizeStream(stream);
    }

    if (ret == ACL_SUCCESS) {
      auto size = GetShapeSize(shape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
      if (ret == ACL_SUCCESS) {
        std::vector<float> expectedData(size, 0);
        ReferenceMul(selfHostData.data(), otherHostData.data(), expectedData.data(), size);
        pass = CompareResult(resultData.data(), expectedData.data(), size,
                             ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
        printf("  Expected: ");
        for (int64_t i = 0; i < size; i++) printf("%.1f ", expectedData[i]);
        printf("\n");
        printf("  Actual:   ");
        for (int64_t i = 0; i < size; i++) printf("%.1f ", resultData[i]);
        printf("\n");
      }
    }
  }

  if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  if (self != nullptr) aclDestroyTensor(self);
  if (other != nullptr) aclDestroyTensor(other);
  if (out != nullptr) aclDestroyTensor(out);
  if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
  if (otherDeviceAddr != nullptr) aclrtFree(otherDeviceAddr);
  if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);

  PrintTestResult("Mul_1D", pass);
  stats.record(pass);
  return pass;
}

// 6.3 3D张量测试
bool TestMul3D(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Mul_3D [2,2,2]*[2,2,2]\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> shape = {2, 2, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> otherHostData = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  std::vector<float> outHostData(8, 0.0f);

  auto ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret == ACL_SUCCESS) {
    ret = CreateAclTensor(otherHostData, shape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  }
  if (ret == ACL_SUCCESS) {
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  }

  if (ret == ACL_SUCCESS) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

    if (ret == ACL_SUCCESS && workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    if (ret == ACL_SUCCESS) {
      ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    }

    if (ret == ACL_SUCCESS) {
      ret = aclrtSynchronizeStream(stream);
    }

    if (ret == ACL_SUCCESS) {
      auto size = GetShapeSize(shape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
      if (ret == ACL_SUCCESS) {
        std::vector<float> expectedData(size, 0);
        ReferenceMul(selfHostData.data(), otherHostData.data(), expectedData.data(), size);
        pass = CompareResult(resultData.data(), expectedData.data(), size,
                             ToleranceConfig::FLOAT32_ATOL, ToleranceConfig::FLOAT32_RTL);
        printf("  Expected: ");
        for (int64_t i = 0; i < size; i++) printf("%.1f ", expectedData[i]);
        printf("\n");
        printf("  Actual:   ");
        for (int64_t i = 0; i < size; i++) printf("%.1f ", resultData[i]);
        printf("\n");
      }
    }
  }

  if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  if (self != nullptr) aclDestroyTensor(self);
  if (other != nullptr) aclDestroyTensor(other);
  if (out != nullptr) aclDestroyTensor(out);
  if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
  if (otherDeviceAddr != nullptr) aclrtFree(otherDeviceAddr);
  if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);

  PrintTestResult("Mul_3D", pass);
  stats.record(pass);
  return pass;
}

// 6.4 INT标量乘法测试
bool TestMulsIntScalar(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Muls_INT_Scalar [4,2]*int(3)\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> shape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> outHostData(8, 0.0f);
  int scalarValue = 3;

  auto ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret == ACL_SUCCESS) {
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  }

  aclScalar* scalar = nullptr;
  if (ret == ACL_SUCCESS) {
    scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_INT32);
  }

  if (ret == ACL_SUCCESS && scalar != nullptr && self != nullptr && out != nullptr) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);

    if (ret == ACL_SUCCESS && workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    if (ret == ACL_SUCCESS) {
      ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
    }

    if (ret == ACL_SUCCESS) {
      ret = aclrtSynchronizeStream(stream);
    }

    if (ret == ACL_SUCCESS) {
      auto size = GetShapeSize(shape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
      if (ret == ACL_SUCCESS) {
        bool allMatch = true;
        for (int64_t i = 0; i < size; i++) {
          float expected = selfHostData[i] * static_cast<float>(scalarValue);
          float diff = std::fabs(resultData[i] - expected);
          float tolerance = ToleranceConfig::FLOAT32_ATOL + ToleranceConfig::FLOAT32_RTL * std::fabs(expected);
          if (diff > tolerance) {
            printf("  Mismatch at %ld: actual=%f, expected=%f\n", i, resultData[i], expected);
            allMatch = false;
            break;
          }
        }
        pass = allMatch;
        printf("  Scalar: %d\n", scalarValue);
        printf("  Expected: ");
        for (int64_t i = 0; i < size; i++) printf("%.1f ", selfHostData[i] * scalarValue);
        printf("\n");
        printf("  Actual:   ");
        for (int64_t i = 0; i < size; i++) printf("%.1f ", resultData[i]);
        printf("\n");
      }
    }
  }

  if (scalar != nullptr) aclDestroyScalar(scalar);
  if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  if (self != nullptr) aclDestroyTensor(self);
  if (out != nullptr) aclDestroyTensor(out);
  if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
  if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);

  PrintTestResult("Muls_INT_Scalar", pass);
  stats.record(pass);
  return pass;
}

// 6.5 标量与1D张量广播测试
bool TestMuls1D(aclrtStream stream, TestStats& stats) {
  printf("\n>>> Running Test: Muls_1D [8]*float(2.5)\n");
  bool pass = false;
  void* workspaceAddr = nullptr;

  std::vector<int64_t> shape = {8};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> outHostData(8, 0.0f);
  float scalarValue = 2.5f;

  auto ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  if (ret == ACL_SUCCESS) {
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  }

  aclScalar* scalar = nullptr;
  if (ret == ACL_SUCCESS) {
    scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
  }

  if (ret == ACL_SUCCESS && scalar != nullptr && self != nullptr && out != nullptr) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnMulsGetWorkspaceSize(self, scalar, out, &workspaceSize, &executor);

    if (ret == ACL_SUCCESS && workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    if (ret == ACL_SUCCESS) {
      ret = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
    }

    if (ret == ACL_SUCCESS) {
      ret = aclrtSynchronizeStream(stream);
    }

    if (ret == ACL_SUCCESS) {
      auto size = GetShapeSize(shape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDeviceAddr,
                        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
      if (ret == ACL_SUCCESS) {
        bool allMatch = true;
        for (int64_t i = 0; i < size; i++) {
          float expected = selfHostData[i] * scalarValue;
          float diff = std::fabs(resultData[i] - expected);
          float tolerance = ToleranceConfig::FLOAT32_ATOL + ToleranceConfig::FLOAT32_RTL * std::fabs(expected);
          if (diff > tolerance) {
            printf("  Mismatch at %ld: actual=%f, expected=%f\n", i, resultData[i], expected);
            allMatch = false;
            break;
          }
        }
        pass = allMatch;
        printf("  Scalar: %.1f\n", scalarValue);
        printf("  Expected: ");
        for (int64_t i = 0; i < size; i++) printf("%.2f ", selfHostData[i] * scalarValue);
        printf("\n");
        printf("  Actual:   ");
        for (int64_t i = 0; i < size; i++) printf("%.2f ", resultData[i]);
        printf("\n");
      }
    }
  }

  if (scalar != nullptr) aclDestroyScalar(scalar);
  if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
  if (self != nullptr) aclDestroyTensor(self);
  if (out != nullptr) aclDestroyTensor(out);
  if (selfDeviceAddr != nullptr) aclrtFree(selfDeviceAddr);
  if (outDeviceAddr != nullptr) aclrtFree(outDeviceAddr);

  PrintTestResult("Muls_1D", pass);
  stats.record(pass);
  return pass;
}


// 6. 新增测试用例 - 用于提高覆盖率
bool TestMulInt32(aclrtStream stream, TestStats& stats);
bool TestMul1D(aclrtStream stream, TestStats& stats);
bool TestMul3D(aclrtStream stream, TestStats& stats);
bool TestMulsIntScalar(aclrtStream stream, TestStats& stats);
bool TestMuls1D(aclrtStream stream, TestStats& stats);

int main() {
  printf("========================================\n");
  printf("  Mul Operator Test Suite - Step 5\n");
  printf("  (Complete: Types + Broadcast + API + Boundary)\n");
  printf("========================================\n");

  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  TestStats stats;

  // Step 1: 基础测试
  TestMulFloat32Basic(stream, stats);

  // Step 2: 数据类型扩展
  TestMulFloat16Basic(stream, stats);
  TestMulMixedFp16Fp32(stream, stats);

  // Step 3: 广播测试
  TestMulBroadcastRow(stream, stats);
  TestMulBroadcastCol(stream, stats);
  TestMulBroadcastMultiDim(stream, stats);

  // Step 4: API变体测试
  TestMulsFloat32(stream, stats);
  TestInplaceMulFloat32(stream, stats);
  TestInplaceMulsFloat32(stream, stats);

  // Step 5: 边界值和异常测试
  TestMulBoundaryLarge(stream, stats);
  TestMulBoundarySmallAndZero(stream, stats);
  TestMulBoundaryNegative(stream, stats);
  TestMulIncompatibleShape(stream, stats);

  // Step 6: 新增测试 - 提高覆盖率
  TestMulInt32(stream, stats);
  TestMul1D(stream, stats);
  TestMul3D(stream, stats);
  TestMulsIntScalar(stream, stats);
  TestMuls1D(stream, stats);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  stats.printSummary();

  return stats.failed > 0 ? 1 : 0;
}


