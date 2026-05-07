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
  }while (0)

// ============================================================
// Utility Functions
// ============================================================

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

// Compute broadcasted shape from two input shapes
std::vector<int64_t> ComputeBroadcastShape(const std::vector<int64_t>& shape1,
                                            const std::vector<int64_t>& shape2) {
  size_t ndim1 = shape1.size();
  size_t ndim2 = shape2.size();
  size_t out_ndim = std::max(ndim1, ndim2);
  std::vector<int64_t> out_shape(out_ndim, 1);
  for (size_t i = 0; i < out_ndim; ++i) {
    int64_t d1 = (i < out_ndim - ndim1) ? 1 : shape1[i - (out_ndim - ndim1)];
    int64_t d2 = (i < out_ndim - ndim2) ? 1 : shape2[i - (out_ndim - ndim2)];
    if (d1 == d2) {
      out_shape[i] = d1;
    } else if (d1 == 1) {
      out_shape[i] = d2;
    } else if (d2 == 1) {
      out_shape[i] = d1;
    }
  }
  return out_shape;
}

// Check if two shapes are broadcast-compatible
bool IsBroadcastable(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2) {
  size_t ndim1 = shape1.size();
  size_t ndim2 = shape2.size();
  size_t out_ndim = std::max(ndim1, ndim2);
  for (size_t i = 0; i < out_ndim; ++i) {
    int64_t d1 = (i < out_ndim - ndim1) ? 1 : shape1[i - (out_ndim - ndim1)];
    int64_t d2 = (i < out_ndim - ndim2) ? 1 : shape2[i - (out_ndim - ndim2)];
    if (d1 != d2 && d1 != 1 && d2 != 1) {
      return false;
    }
  }
  return true;
}

// ============================================================
// Test Infrastructure
// ============================================================

struct TestContext {
  int32_t deviceId;
  aclrtStream stream;
  bool initialized;

  TestContext() : deviceId(0), stream(nullptr), initialized(false) {}

  int Init() {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return -1);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return -1);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return -1);
    initialized = true;
    return 0;
  }

  void Finalize() {
    if (stream) {
      aclrtDestroyStream(stream);
      stream = nullptr;
    }
    if (initialized) {
      aclrtResetDevice(deviceId);
      aclFinalize();
      initialized = false;
    }
  }
};

// RAII wrapper for device memory
struct DeviceBuffer {
  void* ptr;
  DeviceBuffer() : ptr(nullptr) {}
  ~DeviceBuffer() { if (ptr) aclrtFree(ptr); }
};

// RAII wrapper for tensor
struct TensorGuard {
  aclTensor* tensor;
  TensorGuard() : tensor(nullptr) {}
  ~TensorGuard() { if (tensor) aclDestroyTensor(tensor); }
};

// Template function to create tensor on device
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

// Create tensor with raw bytes (for types like fp16, bf16, etc.)
int CreateAclTensorRaw(const void* hostData, size_t byteSize, const std::vector<int64_t>& shape,
                       void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
  auto ret = aclrtMalloc(deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  ret = aclrtMemcpy(*deviceAddr, byteSize, hostData, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

// Run the full Mul pipeline: GetWorkspaceSize -> alloc -> Mul -> sync
int RunMulPipeline(aclTensor* self, aclTensor* other, aclTensor* out,
                   aclrtStream stream, uint64_t* outWorkspaceSize) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  auto ret = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  if (outWorkspaceSize) *outWorkspaceSize = workspaceSize;

  DeviceBuffer workspaceBuf;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceBuf.ptr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnMul(workspaceBuf.ptr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  return 0;
}

// Copy result from device to host and verify
template <typename T>
bool VerifyResult(const std::vector<T>& expected, const std::vector<int64_t>& shape,
                  void* deviceAddr, const char* testName, float tolerance = 1e-5f) {
  auto size = GetShapeSize(shape);
  std::vector<T> resultData(size);
  auto ret = aclrtMemcpy(resultData.data(), size * sizeof(T), deviceAddr,
                         size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] copy result failed. ERROR: %d\n", testName, ret); return false);

  bool pass = true;
  for (int64_t i = 0; i < size; i++) {
    T exp = expected[i];
    T got = resultData[i];
    if constexpr (std::is_floating_point_v<T>) {
      if (std::abs(static_cast<float>(got) - static_cast<float>(exp)) > tolerance) {
        LOG_PRINT("[%s] MISMATCH at [%ld]: expected %f, got %f\n", testName, i, (float)exp, (float)got);
        pass = false;
      }
    } else {
      if (got != exp) {
        LOG_PRINT("[%s] MISMATCH at [%ld]: expected %lld, got %lld\n", testName, i, (long long)exp, (long long)got);
        pass = false;
      }
    }
  }
  if (pass) {
    LOG_PRINT("[%s] PASSED (size=%ld)\n", testName, size);
  }
  return pass;
}

// Generic verify with float output (when output dtype is float but input is different)
bool VerifyResultFloat(const std::vector<float>& expected, const std::vector<int64_t>& shape,
                       void* deviceAddr, const char* testName, float tolerance = 1e-5f) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size);
  auto ret = aclrtMemcpy(resultData.data(), size * sizeof(float), deviceAddr,
                         size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] copy result failed. ERROR: %d\n", testName, ret); return false);

  bool pass = true;
  for (int64_t i = 0; i < (int64_t)size; i++) {
    float exp = expected[i];
    float got = resultData[i];
    if (std::abs(got - exp) > tolerance) {
      LOG_PRINT("[%s] MISMATCH at [%ld]: expected %f, got %f\n", testName, i, exp, got);
      pass = false;
    }
  }
  if (pass) {
    LOG_PRINT("[%s] PASSED (size=%ld)\n", testName, size);
  }
  return pass;
}

// ============================================================
// Test Cases
// ============================================================

// Test 1: Null pointer checks - covers CheckMulNotNull path
int TestNullPointerChecks(TestContext& ctx) {
  LOG_PRINT("\n========== Test 1: Null Pointer Checks ==========\n");
  int errors = 0;

  // Create a valid tensor first for partial null tests
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> shape = {2, 2};
  void* devAddr = nullptr;
  aclTensor* tensor = nullptr;
  auto ret = CreateAclTensor(data, shape, &devAddr, aclDataType::ACL_FLOAT, &tensor);
  CHECK_RET(ret == 0, errors++; return errors);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // Case 1a: self is NULL -> ACL_ERROR_INVALID_PARAM
  aclnnStatus status = aclnnMulGetWorkspaceSize(nullptr, tensor, tensor, &workspaceSize, &executor);
  if (status != ACL_ERROR_INVALID_PARAM) {
    LOG_PRINT("[Test 1a] EXPECTED ACL_ERROR_INVALID_PARAM, got %d\n", status);
    errors++;
  } else {
    LOG_PRINT("[Test 1a] Null self check PASSED\n");
  }

  // Case 1b: other is NULL -> ACL_ERROR_INVALID_PARAM
  status = aclnnMulGetWorkspaceSize(tensor, nullptr, tensor, &workspaceSize, &executor);
  if (status != ACL_ERROR_INVALID_PARAM) {
    LOG_PRINT("[Test 1b] EXPECTED ACL_ERROR_INVALID_PARAM, got %d\n", status);
    errors++;
  } else {
    LOG_PRINT("[Test 1b] Null other check PASSED\n");
  }

  // Case 1c: out is NULL -> ACL_ERROR_INVALID_PARAM
  status = aclnnMulGetWorkspaceSize(tensor, tensor, nullptr, &workspaceSize, &executor);
  if (status != ACL_ERROR_INVALID_PARAM) {
    LOG_PRINT("[Test 1c] EXPECTED ACL_ERROR_INVALID_PARAM, got %d\n", status);
    errors++;
  } else {
    LOG_PRINT("[Test 1c] Null out check PASSED\n");
  }

  // Cleanup
  aclDestroyTensor(tensor);
  aclrtFree(devAddr);

  return errors;
}

// Test 2: FP32 basic multiplication - all positive
int TestFP32AllPositive(TestContext& ctx) {
  LOG_PRINT("\n========== Test 2: FP32 All Positive ==========\n");

  std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> otherData = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> expected = {2.0f, 6.0f, 12.0f, 20.0f, 30.0f, 42.0f};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);

  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "FP32 All Positive");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 3: FP32 positive/negative mixed
int TestFP32MixedSign(TestContext& ctx) {
  LOG_PRINT("\n========== Test 3: FP32 Mixed Sign ==========\n");

  std::vector<float> selfData = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
  std::vector<float> otherData = {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> expected = {-1.0f, -4.0f, -9.0f, -16.0f, -25.0f, -36.0f};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "FP32 Mixed Sign");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 4: FP32 all negative
int TestFP32AllNegative(TestContext& ctx) {
  LOG_PRINT("\n========== Test 4: FP32 All Negative ==========\n");

  std::vector<float> selfData = {-1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> otherData = {-2.0f, -3.0f, -4.0f, -5.0f};
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> expected = {2.0f, 6.0f, 12.0f, 20.0f};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "FP32 All Negative");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 5: FP32 with zeros
int TestFP32WithZeros(TestContext& ctx) {
  LOG_PRINT("\n========== Test 5: FP32 With Zeros ==========\n");

  std::vector<float> selfData = {0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 2.0f};
  std::vector<float> otherData = {5.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f};
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> expected = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "FP32 With Zeros");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 6: 1D tensor
int Test1DShape(TestContext& ctx) {
  LOG_PRINT("\n========== Test 6: 1D Shape ==========\n");

  std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> otherData = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  std::vector<int64_t> shape = {5};
  std::vector<float> expected = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "1D Shape");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 7: 2D tensor (non-square)
int Test2DNonSquare(TestContext& ctx) {
  LOG_PRINT("\n========== Test 7: 2D Non-Square ==========\n");

  std::vector<float> selfData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> otherData = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1};
  std::vector<int64_t> shape = {3, 4};
  std::vector<float> expected = {2, 2, 9, 4, 20, 6, 35, 8, 42, 10, 77, 12};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "2D Non-Square");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 8: 3D tensor
int Test3DShape(TestContext& ctx) {
  LOG_PRINT("\n========== Test 8: 3D Shape ==========\n");

  std::vector<float> selfData = {1,2, 3,4, 5,6, 7,8, 9,10, 11,12};
  std::vector<float> otherData = {1,1, 1,1, 1,1, 1,1, 1,1, 1,1};
  std::vector<int64_t> shape = {2, 3, 2};
  // Expected = same as self since multiplying by 1
  std::vector<float> expected = selfData;

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "3D Shape");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 9: Broadcasting - {1, 3} x {3, 1} -> {3, 3}
int TestBroadcastCross(TestContext& ctx) {
  LOG_PRINT("\n========== Test 9: Broadcasting Cross {1,3} x {3,1} ==========\n");

  std::vector<float> selfData = {1.0f, 2.0f, 3.0f};         // shape {1, 3}
  std::vector<float> otherData = {10.0f, 20.0f, 30.0f};     // shape {3, 1}
  std::vector<int64_t> selfShape = {1, 3};
  std::vector<int64_t> otherShape = {3, 1};
  std::vector<int64_t> outShape = {3, 3};
  // Broadcast result: outer product
  std::vector<float> expected = {
    10.0f, 20.0f, 30.0f,
    20.0f, 40.0f, 60.0f,
    30.0f, 60.0f, 90.0f
  };

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, selfShape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, otherShape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, outShape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, outShape, outDev, "Broadcast Cross");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 10: Broadcasting - scalar-like {1} x {5}
int TestBroadcastScalarLike(TestContext& ctx) {
  LOG_PRINT("\n========== Test 10: Broadcast Scalar-like {1} x {5} ==========\n");

  std::vector<float> selfData = {3.0f};
  std::vector<float> otherData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<int64_t> selfShape = {1};
  std::vector<int64_t> otherShape = {5};
  std::vector<int64_t> outShape = {5};
  std::vector<float> expected = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, selfShape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, otherShape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, outShape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, outShape, outDev, "Broadcast Scalar-like");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 11: Broadcasting - {2, 1, 3} x {1, 4, 3} -> {2, 4, 3}
int TestBroadcast3D(TestContext& ctx) {
  LOG_PRINT("\n========== Test 11: Broadcast 3D {2,1,3} x {1,4,3} ==========\n");

  std::vector<float> selfData = {1,2,3,  4,5,6};  // {2, 1, 3}
  std::vector<float> otherData = {10,20,30,  100,200,300,  1000,2000,3000,  10000,20000,30000};  // {1, 4, 3}
  std::vector<int64_t> selfShape = {2, 1, 3};
  std::vector<int64_t> otherShape = {1, 4, 3};
  std::vector<int64_t> outShape = {2, 4, 3};
  std::vector<float> expected = {
    10,20,30,   100,200,300,   1000,2000,3000,   10000,20000,30000,
    40,100,180, 400,1000,1800, 4000,10000,18000, 40000,100000,180000
  };

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, selfShape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, otherShape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, outShape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, outShape, outDev, "Broadcast 3D");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 12: INT32 multiplication
int TestINT32(TestContext& ctx) {
  LOG_PRINT("\n========== Test 12: INT32 ==========\n");

  std::vector<int32_t> selfData = {1, 2, 3, 4, 5, 6};
  std::vector<int32_t> otherData = {10, 20, 30, 40, 50, 60};
  std::vector<int64_t> shape = {2, 3};
  std::vector<int32_t> expected = {10, 40, 90, 160, 250, 360};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_INT32, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_INT32, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_INT32, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "INT32");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 13: INT8 multiplication
int TestINT8(TestContext& ctx) {
  LOG_PRINT("\n========== Test 13: INT8 ==========\n");

  std::vector<int8_t> selfData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int8_t> otherData = {2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int64_t> shape = {2, 4};
  std::vector<int8_t> expected = {2, 6, 12, 20, 30, 42, 56, 72};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_INT8, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_INT8, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_INT8, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "INT8");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 14: UINT8 multiplication
int TestUINT8(TestContext& ctx) {
  LOG_PRINT("\n========== Test 14: UINT8 ==========\n");

  std::vector<uint8_t> selfData = {1, 2, 3, 4, 5, 6};
  std::vector<uint8_t> otherData = {10, 20, 30, 40, 50, 6};
  std::vector<int64_t> shape = {2, 3};
  std::vector<uint8_t> expected = {10, 40, 90, 160, 250, 36};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_UINT8, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_UINT8, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_UINT8, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "UINT8");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 15: INT64 multiplication
int TestINT64(TestContext& ctx) {
  LOG_PRINT("\n========== Test 15: INT64 ==========\n");

  std::vector<int64_t> selfData = {1000000, 2000000, 3000000, 4000000};
  std::vector<int64_t> otherData = {1000, 2000, 3000, 4000};
  std::vector<int64_t> shape = {2, 2};
  std::vector<int64_t> expected = {1000000000LL, 4000000000LL, 9000000000LL, 16000000000LL};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_INT64, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_INT64, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_INT64, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "INT64");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 16: BOOL multiplication (logical AND)
int TestBOOL(TestContext& ctx) {
  LOG_PRINT("\n========== Test 16: BOOL ==========\n");

  // In CANN, bool multiplication is element-wise logical AND
  std::vector<uint8_t> selfData = {1, 0, 1, 0};  // true, false, true, false
  std::vector<uint8_t> otherData = {1, 1, 0, 0};  // true, true, false, false
  std::vector<int64_t> shape = {2, 2};
  std::vector<uint8_t> expected = {1, 0, 0, 0};  // true&true, false&true, true&false, false&false

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_BOOL, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_BOOL, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_BOOL, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "BOOL");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 17: FP16 (half precision) multiplication
int TestFP16(TestContext& ctx) {
  LOG_PRINT("\n========== Test 17: FP16 ==========\n");

  // FP16: 1 sign bit, 5 exponent bits, 10 mantissa bits
  // Use known FP16 representations for simple values
  // 1.0f in FP16 = 0x3C00, 2.0f = 0x4000, 3.0f = 0x4200, 4.0f = 0x4400
  // 0.5f = 0x3800, 0.25f = 0x3400
  uint16_t selfFp16[] = {0x3C00, 0x4000, 0x4200, 0x4400};  // 1.0, 2.0, 3.0, 4.0
  uint16_t otherFp16[] = {0x4000, 0x3C00, 0x4200, 0x3800}; // 2.0, 1.0, 3.0, 0.5
  std::vector<int64_t> shape = {2, 2};
  // Expected: 2.0, 2.0, 9.0, 2.0 -> 0x4000, 0x4000, 0x4800, 0x4000
  uint16_t expectedFp16[] = {0x4000, 0x4000, 0x4800, 0x4000};
  size_t byteSize = 4 * sizeof(uint16_t);

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensorRaw(selfFp16, byteSize, shape, &selfDev, aclDataType::ACL_FLOAT16, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensorRaw(otherFp16, byteSize, shape, &otherDev, aclDataType::ACL_FLOAT16, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensorRaw(expectedFp16, byteSize, shape, &outDev, aclDataType::ACL_FLOAT16, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);

  // Verify by copying back FP16 data
  std::vector<uint16_t> resultData(4);
  auto cpyRet = aclrtMemcpy(resultData.data(), byteSize, outDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
  bool pass = (err == 0) && (cpyRet == ACL_SUCCESS);
  if (pass) {
    for (int i = 0; i < 4; i++) {
      if (resultData[i] != expectedFp16[i]) {
        LOG_PRINT("[FP16] MISMATCH at [%d]: expected 0x%04X, got 0x%04X\n", i, expectedFp16[i], resultData[i]);
        pass = false;
      }
    }
  }
  if (pass) LOG_PRINT("[FP16] PASSED\n");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 18: BF16 (bfloat16) multiplication
int TestBF16(TestContext& ctx) {
  LOG_PRINT("\n========== Test 18: BF16 ==========\n");

  // BF16: upper 16 bits of FP32
  // 1.0f = 0x3F800000 -> BF16: 0x3F80
  // 2.0f = 0x40000000 -> BF16: 0x4000
  uint16_t selfBf16[] = {0x3F80, 0x4000, 0x4040, 0x4080};  // 1.0, 2.0, 3.0, 4.0
  uint16_t otherBf16[] = {0x4000, 0x3F80, 0x4040, 0x3F00}; // 2.0, 1.0, 3.0, 0.5
  std::vector<int64_t> shape = {2, 2};
  // Expected: 2.0, 2.0, 9.0, 2.0 -> BF16: 0x4000, 0x4000, 0x4800 (actually 0x4110), 0x4000
  uint16_t expectedBf16[] = {0x4000, 0x4000, 0x4110, 0x4000};
  size_t byteSize = 4 * sizeof(uint16_t);

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensorRaw(selfBf16, byteSize, shape, &selfDev, aclDataType::ACL_BF16, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensorRaw(otherBf16, byteSize, shape, &otherDev, aclDataType::ACL_BF16, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensorRaw(expectedBf16, byteSize, shape, &outDev, aclDataType::ACL_BF16, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);

  std::vector<uint16_t> resultData(4);
  auto cpyRet = aclrtMemcpy(resultData.data(), byteSize, outDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
  bool pass = (err == 0) && (cpyRet == ACL_SUCCESS);
  if (pass) {
    for (int i = 0; i < 4; i++) {
      if (resultData[i] != expectedBf16[i]) {
        LOG_PRINT("[BF16] MISMATCH at [%d]: expected 0x%04X, got 0x%04X\n", i, expectedBf16[i], resultData[i]);
        pass = false;
      }
    }
  }
  if (pass) LOG_PRINT("[BF16] PASSED\n");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 19: Mixed dtype - FP16 x FP32 (covers IsMulMixDtypeSupport path)
int TestMixedFP16FP32(TestContext& ctx) {
  LOG_PRINT("\n========== Test 19: Mixed FP16 x FP32 ==========\n");

  // self is FP16, other is FP32 -> mixed dtype path, result promoted to FP32
  uint16_t selfFp16[] = {0x3C00, 0x4000, 0x4200, 0x4400};  // 1.0, 2.0, 3.0, 4.0
  std::vector<float> otherData = {2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> expected = {2.0f, 6.0f, 12.0f, 20.0f};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  size_t fp16Size = 4 * sizeof(uint16_t);
  auto ret = CreateAclTensorRaw(selfFp16, fp16Size, shape, &selfDev, aclDataType::ACL_FLOAT16, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  // Output is FP32 (promoted type)
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResultFloat(expected, shape, outDev, "Mixed FP16xFP32");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 20: Mixed dtype - BF16 x FP32
int TestMixedBF16FP32(TestContext& ctx) {
  LOG_PRINT("\n========== Test 20: Mixed BF16 x FP32 ==========\n");

  uint16_t selfBf16[] = {0x3F80, 0x4000, 0x4040, 0x4080};  // 1.0, 2.0, 3.0, 4.0
  std::vector<float> otherData = {10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> expected = {10.0f, 40.0f, 90.0f, 160.0f};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  size_t bf16Size = 4 * sizeof(uint16_t);
  auto ret = CreateAclTensorRaw(selfBf16, bf16Size, shape, &selfDev, aclDataType::ACL_BF16, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResultFloat(expected, shape, outDev, "Mixed BF16xFP32");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 21: Mixed dtype - FP32 x FP16 (reversed order)
int TestMixedFP32FP16(TestContext& ctx) {
  LOG_PRINT("\n========== Test 21: Mixed FP32 x FP16 ==========\n");

  std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
  uint16_t otherFp16[] = {0x4000, 0x4200, 0x4400, 0x4500};  // 2.0, 3.0, 4.0, 5.0
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> expected = {2.0f, 6.0f, 12.0f, 20.0f};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  size_t fp16Size = 4 * sizeof(uint16_t);
  ret = CreateAclTensorRaw(otherFp16, fp16Size, shape, &otherDev, aclDataType::ACL_FLOAT16, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResultFloat(expected, shape, outDev, "Mixed FP32xFP16");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 22: INT32 x INT32 with broadcasting
int TestINT32Broadcast(TestContext& ctx) {
  LOG_PRINT("\n========== Test 22: INT32 Broadcast ==========\n");

  std::vector<int32_t> selfData = {1, 2, 3};    // {1, 3}
  std::vector<int32_t> otherData = {10, 20, 30}; // {3, 1}
  std::vector<int64_t> selfShape = {1, 3};
  std::vector<int64_t> otherShape = {3, 1};
  std::vector<int64_t> outShape = {3, 3};
  std::vector<int32_t> expected = {10, 20, 30, 20, 40, 60, 30, 60, 90};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, selfShape, &selfDev, aclDataType::ACL_INT32, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, otherShape, &otherDev, aclDataType::ACL_INT32, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, outShape, &outDev, aclDataType::ACL_INT32, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, outShape, outDev, "INT32 Broadcast");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 23: Cross-dtype promotion INT32 x FP32 -> FP32
int TestINT32xFP32Promote(TestContext& ctx) {
  LOG_PRINT("\n========== Test 23: INT32 x FP32 Promotion ==========\n");

  std::vector<int32_t> selfData = {1, 2, 3, 4};
  std::vector<float> otherData = {1.5f, 2.5f, 3.5f, 4.5f};
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> expected = {1.5f, 5.0f, 10.5f, 18.0f};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_INT32, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResultFloat(expected, shape, outDev, "INT32xFP32 Promotion");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 24: BOOL x FP32 promotion
int TestBOOLxFP32Promote(TestContext& ctx) {
  LOG_PRINT("\n========== Test 24: BOOL x FP32 Promotion ==========\n");

  std::vector<uint8_t> selfData = {1, 0, 1, 0};  // bool
  std::vector<float> otherData = {3.14f, 2.72f, 1.41f, 0.58f};
  std::vector<int64_t> shape = {2, 2};
  // bool promoted with FP32 -> FP32, true=1.0, false=0.0
  std::vector<float> expected = {3.14f, 0.0f, 1.41f, 0.0f};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_BOOL, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResultFloat(expected, shape, outDev, "BOOLxFP32 Promotion");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 25: Unsupported dtype check (ACL_STRING should fail)
int TestUnsupportedDtype(TestContext& ctx) {
  LOG_PRINT("\n========== Test 25: Unsupported Dtype ==========\n");

  // Use ACL_STRING which is NOT in the support list
  std::vector<float> data = {1.0f, 2.0f};
  std::vector<int64_t> shape = {2};
  void* devAddr = nullptr;
  aclTensor* tensor = nullptr;

  // Create tensor with wrong dtype descriptor (FLOAT data but STRING type)
  auto ret = CreateAclTensor(data, shape, &devAddr, aclDataType::ACL_FLOAT, &tensor);
  if (ret != 0) return 1;

  // Create a second tensor with unsupported dtype
  // We can't easily create an ACL_STRING tensor, so test with mismatched out dtype
  void* outDev = nullptr;
  aclTensor* out = nullptr;
  ret = CreateAclTensor(data, shape, &outDev, aclDataType::ACL_STRING, &out);
  if (ret != 0) { aclDestroyTensor(tensor); aclrtFree(devAddr); return 1; }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnMulGetWorkspaceSize(tensor, tensor, out, &workspaceSize, &executor);
  // Should fail with ACLNN_ERR_PARAM_INVALID due to unsupported dtype
  bool pass = (status != ACL_SUCCESS);
  if (pass) {
    LOG_PRINT("[Unsupported Dtype] Correctly rejected (status=%d)\n", status);
  } else {
    LOG_PRINT("[Unsupported Dtype] UNEXPECTED success\n");
  }

  aclDestroyTensor(tensor); aclDestroyTensor(out);
  aclrtFree(devAddr); aclrtFree(outDev);
  return !pass;
}

// Test 26: Incompatible broadcast shapes (should fail)
int TestIncompatibleBroadcast(TestContext& ctx) {
  LOG_PRINT("\n========== Test 26: Incompatible Broadcast ==========\n");

  std::vector<float> selfData = {1, 2, 3, 4};  // shape {2, 2}
  std::vector<float> otherData = {1, 2, 3};     // shape {3} - cannot broadcast with {2, 2}
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> otherShape = {3};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, selfShape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, otherShape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;

  // Output shape doesn't matter since broadcast check should fail first
  std::vector<float> dummyOut = {0, 0, 0, 0};
  ret = CreateAclTensor(dummyOut, selfShape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);
  bool pass = (status != ACL_SUCCESS);
  if (pass) {
    LOG_PRINT("[Incompatible Broadcast] Correctly rejected (status=%d)\n", status);
  } else {
    LOG_PRINT("[Incompatible Broadcast] UNEXPECTED success\n");
  }

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return !pass;
}

// Test 27: Empty tensor handling (workspaceSize = 0)
int TestEmptyTensor(TestContext& ctx) {
  LOG_PRINT("\n========== Test 27: Empty Tensor ==========\n");

  std::vector<int64_t> shape = {0, 3};  // empty tensor
  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  // Allocate zero bytes for empty tensors
  auto ret = aclrtMalloc(&selfDev, 1, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) return 1;
  ret = aclrtMalloc(&otherDev, 1, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) return 1;
  ret = aclrtMalloc(&outDev, 1, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) return 1;

  std::vector<int64_t> strides = {3, 1};
  self = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
                         strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                         shape.data(), shape.size(), selfDev);
  other = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
                          strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                          shape.data(), shape.size(), otherDev);
  out = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
                        strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                        shape.data(), shape.size(), outDev);

  uint64_t workspaceSize = UINT64_MAX;  // sentinel value
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

  bool pass = (status == ACL_SUCCESS) && (workspaceSize == 0);
  if (pass) {
    LOG_PRINT("[Empty Tensor] Correctly returned workspaceSize=0\n");
  } else {
    LOG_PRINT("[Empty Tensor] FAILED: status=%d, workspaceSize=%lu\n", status, (unsigned long)workspaceSize);
  }

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return !pass;
}

// Test 28: DOUBLE (FP64) multiplication
int TestDOUBLE(TestContext& ctx) {
  LOG_PRINT("\n========== Test 28: DOUBLE ==========\n");

  std::vector<double> selfData = {1.5, 2.5, 3.5, 4.5};
  std::vector<double> otherData = {2.0, 3.0, 4.0, 5.0};
  std::vector<int64_t> shape = {2, 2};
  std::vector<double> expected = {3.0, 7.5, 14.0, 22.5};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_DOUBLE, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_DOUBLE, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_DOUBLE, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);

  // Verify DOUBLE result
  auto size = GetShapeSize(shape);
  std::vector<double> resultData(size);
  auto cpyRet = aclrtMemcpy(resultData.data(), size * sizeof(double), outDev,
                            size * sizeof(double), ACL_MEMCPY_DEVICE_TO_HOST);
  bool pass = (err == 0) && (cpyRet == ACL_SUCCESS);
  if (pass) {
    for (int64_t i = 0; i < size; i++) {
      if (std::abs(resultData[i] - expected[i]) > 1e-10) {
        LOG_PRINT("[DOUBLE] MISMATCH at [%ld]: expected %f, got %f\n", i, expected[i], resultData[i]);
        pass = false;
      }
    }
  }
  if (pass) LOG_PRINT("[DOUBLE] PASSED\n");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 29: FP32 with larger tensor (64 elements)
int TestLargeTensor(TestContext& ctx) {
  LOG_PRINT("\n========== Test 29: Large Tensor (4x4x4) ==========\n");

  std::vector<float> selfData(64);
  std::vector<float> otherData(64);
  std::vector<float> expected(64);
  for (int i = 0; i < 64; i++) {
    selfData[i] = static_cast<float>(i + 1);
    otherData[i] = static_cast<float>(i + 1);
    expected[i] = static_cast<float>((i + 1) * (i + 1));
  }
  std::vector<int64_t> shape = {4, 4, 4};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "Large Tensor");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 30: INT16 multiplication
int TestINT16(TestContext& ctx) {
  LOG_PRINT("\n========== Test 30: INT16 ==========\n");

  std::vector<int16_t> selfData = {100, 200, 300, 400};
  std::vector<int16_t> otherData = {10, 20, 30, 40};
  std::vector<int64_t> shape = {2, 2};
  std::vector<int16_t> expected = {1000, 4000, 9000, 16000};

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_INT16, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_INT16, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_INT16, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "INT16");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 31: FP32 broadcast {2,1} x {1,3}
int TestBroadcast2D(TestContext& ctx) {
  LOG_PRINT("\n========== Test 31: Broadcast 2D {2,1} x {1,3} ==========\n");

  std::vector<float> selfData = {1.0f, 2.0f};         // shape {2, 1}
  std::vector<float> otherData = {10.0f, 20.0f, 30.0f}; // shape {1, 3}
  std::vector<int64_t> selfShape = {2, 1};
  std::vector<int64_t> otherShape = {1, 3};
  std::vector<int64_t> outShape = {2, 3};
  std::vector<float> expected = {
    10.0f, 20.0f, 30.0f,
    20.0f, 40.0f, 60.0f
  };

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, selfShape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, otherShape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, outShape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, outShape, outDev, "Broadcast 2D");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 32: Workspace size query only (covers the GetWorkspaceSize path without execution)
int TestWorkspaceSizeQuery(TestContext& ctx) {
  LOG_PRINT("\n========== Test 32: Workspace Size Query ==========\n");

  std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> otherData = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<int64_t> shape = {2, 4};
  std::vector<float> outData(8, 0.0f);

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(outData, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnMulGetWorkspaceSize(self, other, out, &workspaceSize, &executor);

  bool pass = (status == ACL_SUCCESS);
  if (pass) {
    LOG_PRINT("[Workspace Query] PASSED (workspaceSize=%lu)\n", (unsigned long)workspaceSize);
  } else {
    LOG_PRINT("[Workspace Query] FAILED (status=%d)\n", status);
  }

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return !pass;
}

// Test 33: 4D tensor
int Test4DShape(TestContext& ctx) {
  LOG_PRINT("\n========== Test 33: 4D Shape ==========\n");

  std::vector<float> selfData = {1,2, 3,4,  5,6, 7,8};   // {2, 2, 2, 1}
  std::vector<float> otherData = {1, 2};                    // {2}
  std::vector<int64_t> selfShape = {2, 2, 2, 1};
  std::vector<int64_t> otherShape = {2};
  std::vector<int64_t> outShape = {2, 2, 2, 2};  // broadcast result
  std::vector<float> expected = {
    1,2,  2,4,   // [0,0,:,:]
    3,6,  4,8,   // [0,1,:,:]
    5,10, 6,12,  // [1,0,:,:]
    7,14, 8,16   // [1,1,:,:]
  };

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, selfShape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, otherShape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, outShape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, outShape, outDev, "4D Shape");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 34: Identity multiplication (all 1.0)
int TestIdentityMul(TestContext& ctx) {
  LOG_PRINT("\n========== Test 34: Identity Multiplication ==========\n");

  std::vector<float> selfData(16, 0.0f);
  for (int i = 0; i < 16; i++) selfData[i] = static_cast<float>(i);
  std::vector<float> otherData(16, 1.0f);  // all ones
  std::vector<int64_t> shape = {4, 4};
  // Expected = same as self since multiplying by 1
  std::vector<float> expected = selfData;

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "Identity Mul");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// Test 35: Zero multiplication (all zeros in one input)
int TestZeroMul(TestContext& ctx) {
  LOG_PRINT("\n========== Test 35: Zero Multiplication ==========\n");

  std::vector<float> selfData(12, 0.0f);  // all zeros
  std::vector<float> otherData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int64_t> shape = {3, 4};
  std::vector<float> expected(12, 0.0f);

  void* selfDev = nullptr, *otherDev = nullptr, *outDev = nullptr;
  aclTensor* self = nullptr, *other = nullptr, *out = nullptr;

  auto ret = CreateAclTensor(selfData, shape, &selfDev, aclDataType::ACL_FLOAT, &self);
  if (ret != 0) return 1;
  ret = CreateAclTensor(otherData, shape, &otherDev, aclDataType::ACL_FLOAT, &other);
  if (ret != 0) return 1;
  ret = CreateAclTensor(expected, shape, &outDev, aclDataType::ACL_FLOAT, &out);
  if (ret != 0) return 1;

  int err = RunMulPipeline(self, other, out, ctx.stream, nullptr);
  bool pass = (err == 0) && VerifyResult(expected, shape, outDev, "Zero Mul");

  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out);
  aclrtFree(selfDev); aclrtFree(otherDev); aclrtFree(outDev);
  return err != 0 || !pass;
}

// ============================================================
// Main Entry
// ============================================================

int main() {
  LOG_PRINT("========================================\n");
  LOG_PRINT("  Mul Operator Comprehensive Test Suite\n");
  LOG_PRINT("========================================\n");

  TestContext ctx;
  int ret = ctx.Init();
  if (ret != 0) {
    LOG_PRINT("FATAL: Failed to initialize ACL context\n");
    return -1;
  }

  int totalErrors = 0;

  // --- Parameter Validation Tests ---
  totalErrors += TestNullPointerChecks(ctx);
  totalErrors += TestUnsupportedDtype(ctx);
  totalErrors += TestIncompatibleBroadcast(ctx);
  totalErrors += TestEmptyTensor(ctx);
  totalErrors += TestWorkspaceSizeQuery(ctx);

  // --- FP32 Tests (various data patterns) ---
  totalErrors += TestFP32AllPositive(ctx);
  totalErrors += TestFP32MixedSign(ctx);
  totalErrors += TestFP32AllNegative(ctx);
  totalErrors += TestFP32WithZeros(ctx);
  totalErrors += TestIdentityMul(ctx);
  totalErrors += TestZeroMul(ctx);

  // --- Shape Tests ---
  totalErrors += Test1DShape(ctx);
  totalErrors += Test2DNonSquare(ctx);
  totalErrors += Test3DShape(ctx);
  totalErrors += Test4DShape(ctx);
  totalErrors += TestLargeTensor(ctx);

  // --- Broadcasting Tests ---
  totalErrors += TestBroadcastCross(ctx);
  totalErrors += TestBroadcastScalarLike(ctx);
  totalErrors += TestBroadcast3D(ctx);
  totalErrors += TestBroadcast2D(ctx);

  // --- Integer Dtype Tests ---
  totalErrors += TestINT32(ctx);
  totalErrors += TestINT8(ctx);
  totalErrors += TestUINT8(ctx);
  totalErrors += TestINT16(ctx);
  totalErrors += TestINT64(ctx);

  // --- Floating Point Dtype Tests ---
  totalErrors += TestFP16(ctx);
  totalErrors += TestBF16(ctx);
  totalErrors += TestDOUBLE(ctx);

  // --- BOOL Test ---
  totalErrors += TestBOOL(ctx);

  // --- Mixed Dtype Tests ---
  totalErrors += TestMixedFP16FP32(ctx);
  totalErrors += TestMixedBF16FP32(ctx);
  totalErrors += TestMixedFP32FP16(ctx);

  // --- Dtype Promotion Tests ---
  totalErrors += TestINT32Broadcast(ctx);
  totalErrors += TestINT32xFP32Promote(ctx);
  totalErrors += TestBOOLxFP32Promote(ctx);

  // --- Summary ---
  LOG_PRINT("\n========================================\n");
  if (totalErrors == 0) {
    LOG_PRINT("  ALL TESTS PASSED!\n");
  } else {
    LOG_PRINT("  %d TEST(S) FAILED!\n", totalErrors);
  }
  LOG_PRINT("========================================\n");

  ctx.Finalize();
  return totalErrors;
}
