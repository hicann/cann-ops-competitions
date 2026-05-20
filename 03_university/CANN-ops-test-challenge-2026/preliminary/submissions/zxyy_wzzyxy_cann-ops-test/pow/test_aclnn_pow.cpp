/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0
 *
 * Pow算子端到端测试用例
 * 覆盖以下执行路径：
 *   1. PowTensorScalar - 7种dtype tiling路径 (fp16/bf16/fp32/uint8/int8/int16/int32)
 *   2. PowTensorScalar - 特殊指数值路径 (Pows: 0.5/2.0/3.0/-0.5/-1.0/-2.0)
 *   3. PowTensorScalar - Square路径 (exponent=2.0, fp32/fp16/bf16)
 *   4. PowTensorScalar - 整数dtype+负指数报错
 *   5. PowTensorScalar - 空tensor提前返回
 *   6. PowTensorScalar - Inplace版本
 *   7. PowScalarTensor - scalar=1.0 fill(1)分支
 *   8. PowScalarTensor - 普通计算路径
 *   9. PowTensorTensor - 7种dtype tiling路径
 *  10. PowTensorTensor - broadcast场景
 *  11. PowTensorTensor - Inplace版本
 *  12. Exp2 - 浮点直接路径
 *  13. Exp2 - 整数/bool先cast到float
 *  14. Exp2 - bf16先cast到float
 *  15. Exp2 - Inplace版本
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"
#include "aclnnop/aclnn_exp2.h"

// ============================================================
// 宏定义
// ============================================================
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

#define TEST_CASE(name) \
  do { printf("\n[TEST] %s\n", name); } while (0)

#define TEST_PASS(name) \
  do { printf("[PASS] %s\n", name); g_passCount++; } while (0)

#define TEST_FAIL(name, err) \
  do { printf("[FAIL] %s, ret=%d\n", name, err); g_failCount++; } while (0)

static int g_passCount = 0;
static int g_failCount = 0;

// ============================================================
// 工具函数
// ============================================================
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
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

// 创建空tensor（shape含0）
int CreateEmptyAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                         aclDataType dataType, aclTensor** tensor) {
  // 空tensor不需要分配内存，deviceAddr为nullptr
  *deviceAddr = nullptr;
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

// 执行算子并同步
int RunWorkspace(uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream,
                 void** workspaceAddr) {
  if (workspaceSize > 0) {
    auto ret = aclrtMalloc(workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  return 0;
}

// 释放tensor和device内存
void FreeTensorAndMem(aclTensor* tensor, void* deviceAddr) {
  if (tensor) aclDestroyTensor(tensor);
  if (deviceAddr) aclrtFree(deviceAddr);
}

// ============================================================
// 半精度辅助（fp16用uint16_t存储）
// ============================================================
// 简单的float->fp16转换（仅用于构造测试数据）
uint16_t FloatToFp16(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint16_t h = (uint16_t)(((x >> 16) & 0x8000) |
               ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
               ((x >> 13) & 0x03ff));
  return h;
}

// fp16 -> float
float Fp16ToFloat(uint16_t h) {
  uint32_t s = (h & 0x8000) << 16;
  uint32_t e = (h & 0x7c00) >> 10;
  uint32_t m = (h & 0x03ff) << 13;
  uint32_t f;
  if (e == 0) {
    if (m != 0) {
      while ((m & 0x00800000) == 0) {
        m <<= 1;
        e--;
      }
      e++;
      m &= ~0x00800000;
    }
  } else if (e == 31) {
    e = 255;
  } else {
    e += 127 - 15;
  }
  f = s | (e << 23) | m;
  float res;
  memcpy(&res, &f, 4);
  return res;
}

// bf16 -> float
float Bf16ToFloat(uint16_t h) {
  uint32_t f = h << 16;
  float res;
  memcpy(&res, &f, 4);
  return res;
}

// ============================================================
// 结果验证工具
// ============================================================

bool AlmostEqual(double expected, double actual, double atol = 1e-4, double rtol = 1e-4) {
  if (std::isnan(expected) && std::isnan(actual)) return true;
  if (std::isinf(expected) && std::isinf(actual)) {
    return (expected > 0) == (actual > 0);
  }
  double diff = std::abs(expected - actual);
  return diff <= (atol + rtol * std::abs(expected));
}

template <typename T>
int VerifyResult(aclTensor* out, void* deviceAddr, const std::vector<double>& expected, const char* caseName) {
  auto shape = std::vector<int64_t>(expected.size()); // 简单起见，这里假设size匹配
  int64_t size = expected.size();
  std::vector<T> hostData(size);

  auto ret = aclrtMemcpy(hostData.data(), size * sizeof(T), deviceAddr, size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy DeviceToHost failed. ERROR: %d\n", ret); return ret);

  bool allPass = true;
  for (int64_t i = 0; i < size; i++) {
    double actualVal = 0;
    // 数据转换
    if (std::is_same<T, uint16_t>::value) {
        // 这里需要区分是fp16还是bf16，暂时根据caseName或者外部传入判断比较好
        // 简单处理：PTS-002是fp16, PTS-003是bf16
        if (strstr(caseName, "float16") || strstr(caseName, "fp16")) {
            actualVal = (double)Fp16ToFloat(*(uint16_t*)&hostData[i]);
        } else {
            actualVal = (double)Bf16ToFloat(*(uint16_t*)&hostData[i]);
        }
    } else {
        actualVal = (double)hostData[i];
    }

    if (!AlmostEqual(expected[i], actualVal)) {
      LOG_PRINT("  [Check Failed] %s: Index %ld, expected %f, actual %f\n", caseName, i, expected[i], actualVal);
      allPass = false;
      break;
    }
  }

  if (allPass) {
    TEST_PASS(caseName);
  } else {
    TEST_FAIL(caseName, -1);
  }
  return allPass ? 0 : -1;
}

// ============================================================
// PowTensorScalar 测试函数
// ============================================================

/**
 * 通用PowTensorScalar执行函数
 * 覆盖路径：参数校验 -> 类型推导 -> 计算路径选择 -> 执行
 */
int RunPowTensorScalar(aclrtStream stream, const char* caseName,
                       aclTensor* self, aclScalar* exponent, aclTensor* out, void* outAddr,
                       const std::vector<double>& expected = {}, bool expectError = false) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  void* workspaceAddr = nullptr;

  auto ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
  if (expectError) {
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("  [Expected error] aclnnPowTensorScalarGetWorkspaceSize returned %d\n", ret);
      TEST_PASS(caseName);
      return 0;
    }
    LOG_PRINT("  [Unexpected success] Expected error but got ACL_SUCCESS\n");
    TEST_FAIL(caseName, -1);
    return -1;
  }
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclnnPowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret);
    TEST_FAIL(caseName, ret); return ret);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("  allocate workspace failed. ERROR: %d\n", ret);
      TEST_FAIL(caseName, ret); return ret);
  }

  ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclnnPowTensorScalar failed. ERROR: %d\n", ret);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    TEST_FAIL(caseName, ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    TEST_FAIL(caseName, ret); return ret);

  if (workspaceAddr) aclrtFree(workspaceAddr);

  if (!expected.empty()) {
    // 根据caseName或者tensor类型动态选择VerifyResult模板类型
    // 这里简单通过caseName包含的信息决定，生产代码通常从aclDataType获取
    if (strstr(caseName, "float32")) return VerifyResult<float>(out, outAddr, expected, caseName);
    if (strstr(caseName, "float16") || strstr(caseName, "fp16") || strstr(caseName, "bf16"))
        return VerifyResult<uint16_t>(out, outAddr, expected, caseName);
    if (strstr(caseName, "uint8")) return VerifyResult<uint8_t>(out, outAddr, expected, caseName);
    if (strstr(caseName, "int8")) return VerifyResult<int8_t>(out, outAddr, expected, caseName);
    if (strstr(caseName, "int16")) return VerifyResult<int16_t>(out, outAddr, expected, caseName);
    if (strstr(caseName, "int32")) return VerifyResult<int32_t>(out, outAddr, expected, caseName);
    return VerifyResult<float>(out, outAddr, expected, caseName); // 默认float
  }

  TEST_PASS(caseName);
  return 0;
}

/**
 * 通用InplacePowTensorScalar执行函数
 */
int RunInplacePowTensorScalar(aclrtStream stream, const char* caseName,
                               aclTensor* self, aclScalar* exponent) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  void* workspaceAddr = nullptr;

  auto ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self, exponent, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclnnInplacePowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret);
    TEST_FAIL(caseName, ret); return ret);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("  allocate workspace failed. ERROR: %d\n", ret);
      TEST_FAIL(caseName, ret); return ret);
  }

  ret = aclnnInplacePowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclnnInplacePowTensorScalar failed. ERROR: %d\n", ret);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    TEST_FAIL(caseName, ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    TEST_FAIL(caseName, ret); return ret);

  if (workspaceAddr) aclrtFree(workspaceAddr);
  TEST_PASS(caseName);
  return 0;
}

// ============================================================
// PowTensorScalar 各dtype tiling路径测试
// ============================================================

/**
 * TC-PTS-001: float32 tensor + 普通指数 (通用Pow路径, tilingKey=3001)
 */
void TestPowTensorScalar_Float32_NormalExp(aclrtStream stream) {
  TEST_CASE("TC-PTS-001: PowTensorScalar float32 normal exponent (Pow path, tilingKey=3001)");
  std::vector<int64_t> shape = {4, 4};
  std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 1.5f, 2.5f, 3.5f,
                                  0.1f, 0.2f, 0.3f, 0.4f, 1.1f, 1.2f, 1.3f, 1.4f};
  std::vector<float> outData(16, 0.0f);
  float expVal = 4.1f;

  // 计算期望值
  std::vector<double> expected(16);
  for (int i = 0; i < 16; i++) {
    expected[i] = std::pow((double)selfData[i], (double)expVal);
  }

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  RunPowTensorScalar(stream, "TC-PTS-001_float32", self, exponent, out, outAddr, expected);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-002: float16 tensor + 普通指数 (tilingKey=1001)
 */
void TestPowTensorScalar_Float16_NormalExp(aclrtStream stream) {
  TEST_CASE("TC-PTS-002: PowTensorScalar float16 normal exponent (tilingKey=1001)");
  std::vector<int64_t> shape = {8};
  // fp16数据用uint16_t存储
  std::vector<uint16_t> selfData = {
    FloatToFp16(1.0f), FloatToFp16(2.0f), FloatToFp16(3.0f), FloatToFp16(4.0f),
    FloatToFp16(0.5f), FloatToFp16(1.5f), FloatToFp16(2.5f), FloatToFp16(3.5f)
  };
  std::vector<uint16_t> outData(8, 0);
  float expVal = 3.0f;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT16, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT16, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  RunPowTensorScalar(stream, "TC-PTS-002", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-003: bfloat16 tensor + 普通指数 (tilingKey=2001)
 */
void TestPowTensorScalar_BFloat16_NormalExp(aclrtStream stream) {
  TEST_CASE("TC-PTS-003: PowTensorScalar bfloat16 normal exponent (tilingKey=2001)");
  std::vector<int64_t> shape = {8};
  // bf16用uint16_t存储（取float的高16位）
  std::vector<uint16_t> selfData(8);
  float vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 1.5f, 2.5f, 3.5f};
  for (int i = 0; i < 8; i++) {
    uint32_t x; memcpy(&x, &vals[i], 4);
    selfData[i] = (uint16_t)(x >> 16);
  }
  std::vector<uint16_t> outData(8, 0);
  float expVal = 3.0f;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_BF16, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_BF16, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  RunPowTensorScalar(stream, "TC-PTS-003", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-004: uint8 tensor + 正整数指数 (tilingKey=4001)
 */
void TestPowTensorScalar_Uint8_PosExp(aclrtStream stream) {
  TEST_CASE("TC-PTS-004: PowTensorScalar uint8 positive exponent (tilingKey=4001)");
  std::vector<int64_t> shape = {8};
  std::vector<uint8_t> selfData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint8_t> outData(8, 0);
  int64_t expVal = 2;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_UINT8, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_UINT8, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_INT64);

  RunPowTensorScalar(stream, "TC-PTS-004", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-005: int8 tensor + 正整数指数 (tilingKey=5001)
 */
void TestPowTensorScalar_Int8_PosExp(aclrtStream stream) {
  TEST_CASE("TC-PTS-005: PowTensorScalar int8 positive exponent (tilingKey=5001)");
  std::vector<int64_t> shape = {8};
  std::vector<int8_t> selfData = {-3, -2, -1, 0, 1, 2, 3, 4};
  std::vector<int8_t> outData(8, 0);
  int64_t expVal = 3;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_INT8, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_INT8, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_INT64);

  RunPowTensorScalar(stream, "TC-PTS-005", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-006: int16 tensor + 正整数指数 (tilingKey=6001)
 */
void TestPowTensorScalar_Int16_PosExp(aclrtStream stream) {
  TEST_CASE("TC-PTS-006: PowTensorScalar int16 positive exponent (tilingKey=6001)");
  std::vector<int64_t> shape = {8};
  std::vector<int16_t> selfData = {-3, -2, -1, 0, 1, 2, 3, 4};
  std::vector<int16_t> outData(8, 0);
  int64_t expVal = 2;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_INT16, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_INT16, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_INT64);

  RunPowTensorScalar(stream, "TC-PTS-006", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-007: int32 tensor + 正整数指数 (tilingKey=7001)
 */
void TestPowTensorScalar_Int32_PosExp(aclrtStream stream) {
  TEST_CASE("TC-PTS-007: PowTensorScalar int32 positive exponent (tilingKey=7001)");
  std::vector<int64_t> shape = {8};
  std::vector<int32_t> selfData = {-3, -2, -1, 0, 1, 2, 3, 4};
  std::vector<int32_t> outData(8, 0);
  int64_t expVal = 3;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_INT32, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_INT32, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_INT64);

  RunPowTensorScalar(stream, "TC-PTS-007", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

// ============================================================
// PowTensorScalar 特殊指数值路径 (CheckSupportPows)
// ============================================================

/**
 * TC-PTS-008: float32 + exponent=0.5 (Pows路径: sqrt)
 */
void TestPowTensorScalar_Float32_Sqrt(aclrtStream stream) {
  TEST_CASE("TC-PTS-008: PowTensorScalar float32 exponent=0.5 (Pows/sqrt path)");
  std::vector<int64_t> shape = {8};
  std::vector<float> selfData = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f};
  std::vector<float> outData(8, 0.0f);
  float expVal = 0.5f;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  RunPowTensorScalar(stream, "TC-PTS-008", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-009: float32 + exponent=2.0 (Square路径)
 */
void TestPowTensorScalar_Float32_Square(aclrtStream stream) {
  TEST_CASE("TC-PTS-009: PowTensorScalar float32 exponent=2.0 (Square path)");
  std::vector<int64_t> shape = {8};
  std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, 0.5f, -0.5f};
  std::vector<float> outData(8, 0.0f);
  float expVal = 2.0f;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  RunPowTensorScalar(stream, "TC-PTS-009", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-010: float32 + exponent=3.0 (Pows路径: cube)
 */
void TestPowTensorScalar_Float32_Cube(aclrtStream stream) {
  TEST_CASE("TC-PTS-010: PowTensorScalar float32 exponent=3.0 (Pows/cube path)");
  std::vector<int64_t> shape = {8};
  std::vector<float> selfData = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 0.5f, -0.5f, 0.0f};
  std::vector<float> outData(8, 0.0f);
  float expVal = 3.0f;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  RunPowTensorScalar(stream, "TC-PTS-010", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-011: float32 + exponent=-0.5 (Pows路径: rsqrt)
 */
void TestPowTensorScalar_Float32_NegSqrt(aclrtStream stream) {
  TEST_CASE("TC-PTS-011: PowTensorScalar float32 exponent=-0.5 (Pows/rsqrt path)");
  std::vector<int64_t> shape = {8};
  std::vector<float> selfData = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f};
  std::vector<float> outData(8, 0.0f);
  float expVal = -0.5f;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  RunPowTensorScalar(stream, "TC-PTS-011", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-012: float32 + exponent=-1.0 (Pows路径: reciprocal)
 */
void TestPowTensorScalar_Float32_NegOne(aclrtStream stream) {
  TEST_CASE("TC-PTS-012: PowTensorScalar float32 exponent=-1.0 (Pows/reciprocal path)");
  std::vector<int64_t> shape = {8};
  std::vector<float> selfData = {1.0f, 2.0f, 4.0f, 8.0f, 0.5f, 0.25f, 0.125f, 10.0f};
  std::vector<float> outData(8, 0.0f);
  float expVal = -1.0f;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  RunPowTensorScalar(stream, "TC-PTS-012", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-013: float32 + exponent=-2.0 (Pows路径)
 */
void TestPowTensorScalar_Float32_NegSquare(aclrtStream stream) {
  TEST_CASE("TC-PTS-013: PowTensorScalar float32 exponent=-2.0 (Pows path)");
  std::vector<int64_t> shape = {8};
  std::vector<float> selfData = {1.0f, 2.0f, 4.0f, 8.0f, 0.5f, 0.25f, 0.125f, 10.0f};
  std::vector<float> outData(8, 0.0f);
  float expVal = -2.0f;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  RunPowTensorScalar(stream, "TC-PTS-013", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-014: float16 + exponent=2.0 (Square路径, fp16)
 */
void TestPowTensorScalar_Float16_Square(aclrtStream stream) {
  TEST_CASE("TC-PTS-014: PowTensorScalar float16 exponent=2.0 (Square path)");
  std::vector<int64_t> shape = {8};
  std::vector<uint16_t> selfData = {
    FloatToFp16(1.0f), FloatToFp16(2.0f), FloatToFp16(3.0f), FloatToFp16(4.0f),
    FloatToFp16(-1.0f), FloatToFp16(-2.0f), FloatToFp16(0.5f), FloatToFp16(-0.5f)
  };
  std::vector<uint16_t> outData(8, 0);
  float expVal = 2.0f;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT16, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT16, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  RunPowTensorScalar(stream, "TC-PTS-014", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

// ============================================================
// PowTensorScalar 边界/错误路径测试
// ============================================================

/**
 * TC-PTS-015: 整数dtype + 负指数 -> 应报错 (CheckPowTensorScalarExponet)
 */
void TestPowTensorScalar_Int32_NegExp_Error(aclrtStream stream) {
  TEST_CASE("TC-PTS-015: PowTensorScalar int32 negative exponent -> error");
  std::vector<int64_t> shape = {4};
  std::vector<int32_t> selfData = {1, 2, 3, 4};
  std::vector<int32_t> outData(4, 0);
  int64_t expVal = -1;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_INT32, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_INT32, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_INT64);

  RunPowTensorScalar(stream, "TC-PTS-015", self, exponent, out, outAddr, {}, /*expectError=*/true);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-016: 空tensor (shape含0) -> 提前返回，workspaceSize=0
 */
void TestPowTensorScalar_EmptyTensor(aclrtStream stream) {
  TEST_CASE("TC-PTS-016: PowTensorScalar empty tensor (IsEmpty path)");
  std::vector<int64_t> shape = {0, 4};  // 空tensor
  void* selfAddr = nullptr;
  void* outAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  CreateEmptyAclTensor(shape, &selfAddr, ACL_FLOAT, &self);
  CreateEmptyAclTensor(shape, &outAddr, ACL_FLOAT, &out);
  float expVal = 2.0f;
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS && workspaceSize == 0) {
    LOG_PRINT("  Empty tensor: workspaceSize=0 as expected\n");
    TEST_PASS("TC-PTS-016");
  } else {
    LOG_PRINT("  Empty tensor: ret=%d, workspaceSize=%lu\n", ret, workspaceSize);
    TEST_FAIL("TC-PTS-016", ret);
  }

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-017: Inplace PowTensorScalar (float32, exponent=3.0)
 */
void TestInplacePowTensorScalar_Float32(aclrtStream stream) {
  TEST_CASE("TC-PTS-017: InplacePowTensorScalar float32 exponent=3.0");
  std::vector<int64_t> shape = {8};
  std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, 0.5f, -0.5f};
  float expVal = 3.0f;

  void* selfAddr = nullptr;
  aclTensor* self = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  RunInplacePowTensorScalar(stream, "TC-PTS-017", self, exponent);

  FreeTensorAndMem(self, selfAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-018: Inplace PowTensorScalar (float16, exponent=2.0, Square路径)
 */
void TestInplacePowTensorScalar_Float16_Square(aclrtStream stream) {
  TEST_CASE("TC-PTS-018: InplacePowTensorScalar float16 exponent=2.0 (Square path)");
  std::vector<int64_t> shape = {8};
  std::vector<uint16_t> selfData = {
    FloatToFp16(1.0f), FloatToFp16(2.0f), FloatToFp16(3.0f), FloatToFp16(4.0f),
    FloatToFp16(-1.0f), FloatToFp16(-2.0f), FloatToFp16(0.5f), FloatToFp16(-0.5f)
  };
  float expVal = 2.0f;

  void* selfAddr = nullptr;
  aclTensor* self = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT16, &self);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  RunInplacePowTensorScalar(stream, "TC-PTS-018", self, exponent);

  FreeTensorAndMem(self, selfAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-019: float32 + exponent=0 (任意数的0次方=1)
 */
void TestPowTensorScalar_Float32_ZeroExp(aclrtStream stream) {
  TEST_CASE("TC-PTS-019: PowTensorScalar float32 exponent=0 (x^0=1)");
  std::vector<int64_t> shape = {8};
  std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, 0.5f, 0.0f};
  std::vector<float> outData(8, 0.0f);
  float expVal = 0.0f;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  RunPowTensorScalar(stream, "TC-PTS-019", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-020: 大shape float32 (触发多核tiling, blockNum>1)
 */
void TestPowTensorScalar_Float32_LargeShape(aclrtStream stream) {
  TEST_CASE("TC-PTS-020: PowTensorScalar float32 large shape (multi-core tiling)");
  std::vector<int64_t> shape = {1024, 1024};
  int64_t n = 1024 * 1024;
  std::vector<float> selfData(n, 2.0f);
  std::vector<float> outData(n, 0.0f);
  float expVal = 3.0f;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  RunPowTensorScalar(stream, "TC-PTS-020", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-021: int32 + exponent=0 (整数0次方=1, 不触发负指数报错)
 */
void TestPowTensorScalar_Int32_ZeroExp(aclrtStream stream) {
  TEST_CASE("TC-PTS-021: PowTensorScalar int32 exponent=0 (valid, x^0=1)");
  std::vector<int64_t> shape = {8};
  std::vector<int32_t> selfData = {-3, -2, -1, 0, 1, 2, 3, 4};
  std::vector<int32_t> outData(8, 0);
  int64_t expVal = 0;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_INT32, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_INT32, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_INT64);

  RunPowTensorScalar(stream, "TC-PTS-021", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

/**
 * TC-PTS-022: float32 多维shape (4D tensor)
 */
void TestPowTensorScalar_Float32_4D(aclrtStream stream) {
  TEST_CASE("TC-PTS-022: PowTensorScalar float32 4D shape");
  std::vector<int64_t> shape = {2, 3, 4, 4};
  int64_t n = 2 * 3 * 4 * 4;
  std::vector<float> selfData(n, 2.0f);
  std::vector<float> outData(n, 0.0f);
  float expVal = 4.1f;

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);

  RunPowTensorScalar(stream, "TC-PTS-022", self, exponent, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(exponent);
}

// ============================================================
// PowScalarTensor 测试
// ============================================================

/**
 * 通用PowScalarTensor执行函数
 */
int RunPowScalarTensor(aclrtStream stream, const char* caseName,
                       aclScalar* self, aclTensor* exponent, aclTensor* out, void* outAddr,
                       const std::vector<double>& expected = {}, bool expectError = false) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  void* workspaceAddr = nullptr;

  auto ret = aclnnPowScalarTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
  if (expectError) {
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("  [Expected error] aclnnPowScalarTensorGetWorkspaceSize returned %d\n", ret);
      TEST_PASS(caseName);
      return 0;
    }
    LOG_PRINT("  [Unexpected success] Expected error but got ACL_SUCCESS\n");
    TEST_FAIL(caseName, -1);
    return -1;
  }
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclnnPowScalarTensorGetWorkspaceSize failed. ERROR: %d\n", ret);
    TEST_FAIL(caseName, ret); return ret);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("  allocate workspace failed. ERROR: %d\n", ret);
      TEST_FAIL(caseName, ret); return ret);
  }

  ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclnnPowScalarTensor failed. ERROR: %d\n", ret);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    TEST_FAIL(caseName, ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    TEST_FAIL(caseName, ret); return ret);

  if (workspaceAddr) aclrtFree(workspaceAddr);

  if (!expected.empty()) {
    if (strstr(caseName, "float32")) return VerifyResult<float>(out, outAddr, expected, caseName);
    if (strstr(caseName, "float16") || strstr(caseName, "fp16") || strstr(caseName, "bf16"))
        return VerifyResult<uint16_t>(out, outAddr, expected, caseName);
    if (strstr(caseName, "int32")) return VerifyResult<int32_t>(out, outAddr, expected, caseName);
    return VerifyResult<float>(out, outAddr, expected, caseName);
  }

  TEST_PASS(caseName);
  return 0;
}

/**
 * TC-PST-001: scalar=1.0 -> fill(1)分支 (IsRegBase路径)
 */
void TestPowScalarTensor_ScalarOne_Fill(aclrtStream stream) {
  TEST_CASE("TC-PST-001: PowScalarTensor scalar=1.0 (fill(1) branch)");
  std::vector<int64_t> shape = {8};
  std::vector<float> expData = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, 0.5f, -0.5f};
  std::vector<float> outData(8, 0.0f);
  float selfVal = 1.0f;

  std::vector<double> expected(8);
  for (int i = 0; i < 8; i++) {
    expected[i] = std::pow((double)selfVal, (double)expData[i]);
  }

  void *expAddr = nullptr, *outAddr = nullptr;
  aclTensor *exponent = nullptr, *out = nullptr;
  CreateAclTensor(expData, shape, &expAddr, ACL_FLOAT, &exponent);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  aclScalar* self = aclCreateScalar(&selfVal, ACL_FLOAT);

  RunPowScalarTensor(stream, "TC-PST-001_float32", self, exponent, out, outAddr, expected);

  FreeTensorAndMem(exponent, expAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(self);
}

/**
 * TC-PST-002: scalar=2.0 + float32 exponent tensor (普通计算路径)
 */
void TestPowScalarTensor_Float32_Normal(aclrtStream stream) {
  TEST_CASE("TC-PST-002: PowScalarTensor scalar=2.0 float32 exponent (normal path)");
  std::vector<int64_t> shape = {8};
  std::vector<float> expData = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<float> outData(8, 0.0f);
  float selfVal = 2.0f;

  void *expAddr = nullptr, *outAddr = nullptr;
  aclTensor *exponent = nullptr, *out = nullptr;
  CreateAclTensor(expData, shape, &expAddr, ACL_FLOAT, &exponent);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);
  aclScalar* self = aclCreateScalar(&selfVal, ACL_FLOAT);

  RunPowScalarTensor(stream, "TC-PST-002", self, exponent, out, outAddr);

  FreeTensorAndMem(exponent, expAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(self);
}

/**
 * TC-PST-003: scalar=3.0 + float16 exponent tensor
 */
void TestPowScalarTensor_Float16_Exp(aclrtStream stream) {
  TEST_CASE("TC-PST-003: PowScalarTensor scalar=3.0 float16 exponent");
  std::vector<int64_t> shape = {8};
  std::vector<uint16_t> expData = {
    FloatToFp16(0.0f), FloatToFp16(1.0f), FloatToFp16(2.0f), FloatToFp16(3.0f),
    FloatToFp16(4.0f), FloatToFp16(5.0f), FloatToFp16(6.0f), FloatToFp16(7.0f)
  };
  std::vector<uint16_t> outData(8, 0);
  float selfVal = 3.0f;

  void *expAddr = nullptr, *outAddr = nullptr;
  aclTensor *exponent = nullptr, *out = nullptr;
  CreateAclTensor(expData, shape, &expAddr, ACL_FLOAT16, &exponent);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT16, &out);
  aclScalar* self = aclCreateScalar(&selfVal, ACL_FLOAT);

  RunPowScalarTensor(stream, "TC-PST-003", self, exponent, out, outAddr);

  FreeTensorAndMem(exponent, expAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(self);
}

/**
 * TC-PST-004: scalar=2.0 + 空exponent tensor -> 提前返回
 */
void TestPowScalarTensor_EmptyExponent(aclrtStream stream) {
  TEST_CASE("TC-PST-004: PowScalarTensor empty exponent tensor (IsEmpty path)");
  std::vector<int64_t> shape = {0};
  void* expAddr = nullptr;
  void* outAddr = nullptr;
  aclTensor* exponent = nullptr;
  aclTensor* out = nullptr;
  CreateEmptyAclTensor(shape, &expAddr, ACL_FLOAT, &exponent);
  CreateEmptyAclTensor(shape, &outAddr, ACL_FLOAT, &out);
  float selfVal = 2.0f;
  aclScalar* self = aclCreateScalar(&selfVal, ACL_FLOAT);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowScalarTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS && workspaceSize == 0) {
    LOG_PRINT("  Empty exponent: workspaceSize=0 as expected\n");
    TEST_PASS("TC-PST-004");
  } else {
    LOG_PRINT("  Empty exponent: ret=%d, workspaceSize=%lu\n", ret, workspaceSize);
    TEST_FAIL("TC-PST-004", ret);
  }

  FreeTensorAndMem(exponent, expAddr);
  FreeTensorAndMem(out, outAddr);
  aclDestroyScalar(self);
}

// ============================================================
// PowTensorTensor 测试
// ============================================================

/**
 * 通用PowTensorTensor执行函数
 */
int RunPowTensorTensor(aclrtStream stream, const char* caseName,
                       aclTensor* self, aclTensor* exponent, aclTensor* out, void* outAddr,
                       const std::vector<double>& expected = {}, bool expectError = false) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  void* workspaceAddr = nullptr;

  auto ret = aclnnPowTensorTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
  if (expectError) {
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("  [Expected error] aclnnPowTensorTensorGetWorkspaceSize returned %d\n", ret);
      TEST_PASS(caseName);
      return 0;
    }
    LOG_PRINT("  [Unexpected success] Expected error but got ACL_SUCCESS\n");
    TEST_FAIL(caseName, -1);
    return -1;
  }
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclnnPowTensorTensorGetWorkspaceSize failed. ERROR: %d\n", ret);
    TEST_FAIL(caseName, ret); return ret);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("  allocate workspace failed. ERROR: %d\n", ret);
      TEST_FAIL(caseName, ret); return ret);
  }

  ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclnnPowTensorTensor failed. ERROR: %d\n", ret);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    TEST_FAIL(caseName, ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    TEST_FAIL(caseName, ret); return ret);

  if (workspaceAddr) aclrtFree(workspaceAddr);

  if (!expected.empty()) {
    if (strstr(caseName, "float32")) return VerifyResult<float>(out, outAddr, expected, caseName);
    if (strstr(caseName, "float16") || strstr(caseName, "fp16") || strstr(caseName, "bf16"))
        return VerifyResult<uint16_t>(out, outAddr, expected, caseName);
    if (strstr(caseName, "uint8")) return VerifyResult<uint8_t>(out, outAddr, expected, caseName);
    if (strstr(caseName, "int8")) return VerifyResult<int8_t>(out, outAddr, expected, caseName);
    if (strstr(caseName, "int16")) return VerifyResult<int16_t>(out, outAddr, expected, caseName);
    if (strstr(caseName, "int32")) return VerifyResult<int32_t>(out, outAddr, expected, caseName);
    return VerifyResult<float>(out, outAddr, expected, caseName);
  }

  TEST_PASS(caseName);
  return 0;
}

/**
 * 通用InplacePowTensorTensor执行函数
 */
int RunInplacePowTensorTensor(aclrtStream stream, const char* caseName,
                               aclTensor* self, aclTensor* exponent) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  void* workspaceAddr = nullptr;

  auto ret = aclnnInplacePowTensorTensorGetWorkspaceSize(self, exponent, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclnnInplacePowTensorTensorGetWorkspaceSize failed. ERROR: %d\n", ret);
    TEST_FAIL(caseName, ret); return ret);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("  allocate workspace failed. ERROR: %d\n", ret);
      TEST_FAIL(caseName, ret); return ret);
  }

  ret = aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclnnInplacePowTensorTensor failed. ERROR: %d\n", ret);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    TEST_FAIL(caseName, ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    TEST_FAIL(caseName, ret); return ret);

  if (workspaceAddr) aclrtFree(workspaceAddr);
  TEST_PASS(caseName);
  return 0;
}

/**
 * TC-PTT-001: float32 x float32 -> float32 (opKey=3, tilingKey=3xxxxx)
 */
void TestPowTensorTensor_Float32(aclrtStream stream) {
  TEST_CASE("TC-PTT-001: PowTensorTensor float32 x float32 (opKey=3)");
  std::vector<int64_t> shape = {4, 2};
  std::vector<float> selfData = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<float> expData  = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f};
  std::vector<float> outData(8, 0.0f);

  std::vector<double> expected(8);
  for (int i = 0; i < 8; i++) {
    expected[i] = std::pow((double)selfData[i], (double)expData[i]);
  }

  void *selfAddr = nullptr, *expAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *exp = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(expData, shape, &expAddr, ACL_FLOAT, &exp);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);

  RunPowTensorTensor(stream, "TC-PTT-001_float32", self, exp, out, outAddr, expected);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(exp, expAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-PTT-002: float16 x float16 -> float16 (opKey=1)
 */
void TestPowTensorTensor_Float16(aclrtStream stream) {
  TEST_CASE("TC-PTT-002: PowTensorTensor float16 x float16 (opKey=1)");
  std::vector<int64_t> shape = {8};
  std::vector<uint16_t> selfData = {
    FloatToFp16(1.0f), FloatToFp16(2.0f), FloatToFp16(3.0f), FloatToFp16(4.0f),
    FloatToFp16(0.5f), FloatToFp16(1.5f), FloatToFp16(2.5f), FloatToFp16(3.5f)
  };
  std::vector<uint16_t> expData = {
    FloatToFp16(2.0f), FloatToFp16(2.0f), FloatToFp16(2.0f), FloatToFp16(2.0f),
    FloatToFp16(3.0f), FloatToFp16(3.0f), FloatToFp16(3.0f), FloatToFp16(3.0f)
  };
  std::vector<uint16_t> outData(8, 0);

  void *selfAddr = nullptr, *expAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *exp = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT16, &self);
  CreateAclTensor(expData, shape, &expAddr, ACL_FLOAT16, &exp);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT16, &out);

  RunPowTensorTensor(stream, "TC-PTT-002", self, exp, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(exp, expAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-PTT-003: bfloat16 x bfloat16 -> bfloat16 (opKey=2)
 */
void TestPowTensorTensor_BFloat16(aclrtStream stream) {
  TEST_CASE("TC-PTT-003: PowTensorTensor bfloat16 x bfloat16 (opKey=2)");
  std::vector<int64_t> shape = {8};
  std::vector<uint16_t> selfData(8), expData(8);
  float svals[] = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 1.5f, 2.5f, 3.5f};
  float evals[] = {2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f};
  for (int i = 0; i < 8; i++) {
    uint32_t x; memcpy(&x, &svals[i], 4); selfData[i] = (uint16_t)(x >> 16);
    memcpy(&x, &evals[i], 4); expData[i] = (uint16_t)(x >> 16);
  }
  std::vector<uint16_t> outData(8, 0);

  void *selfAddr = nullptr, *expAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *exp = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_BF16, &self);
  CreateAclTensor(expData, shape, &expAddr, ACL_BF16, &exp);
  CreateAclTensor(outData, shape, &outAddr, ACL_BF16, &out);

  RunPowTensorTensor(stream, "TC-PTT-003", self, exp, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(exp, expAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-PTT-004: uint8 x uint8 -> uint8 (opKey=4)
 */
void TestPowTensorTensor_Uint8(aclrtStream stream) {
  TEST_CASE("TC-PTT-004: PowTensorTensor uint8 x uint8 (opKey=4)");
  std::vector<int64_t> shape = {8};
  std::vector<uint8_t> selfData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint8_t> expData  = {2, 2, 2, 2, 1, 1, 1, 1};
  std::vector<uint8_t> outData(8, 0);

  void *selfAddr = nullptr, *expAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *exp = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_UINT8, &self);
  CreateAclTensor(expData, shape, &expAddr, ACL_UINT8, &exp);
  CreateAclTensor(outData, shape, &outAddr, ACL_UINT8, &out);

  RunPowTensorTensor(stream, "TC-PTT-004", self, exp, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(exp, expAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-PTT-005: int8 x int8 -> int8 (opKey=5)
 */
void TestPowTensorTensor_Int8(aclrtStream stream) {
  TEST_CASE("TC-PTT-005: PowTensorTensor int8 x int8 (opKey=5)");
  std::vector<int64_t> shape = {8};
  std::vector<int8_t> selfData = {-3, -2, -1, 0, 1, 2, 3, 4};
  std::vector<int8_t> expData  = {3, 3, 3, 3, 2, 2, 2, 2};
  std::vector<int8_t> outData(8, 0);

  void *selfAddr = nullptr, *expAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *exp = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_INT8, &self);
  CreateAclTensor(expData, shape, &expAddr, ACL_INT8, &exp);
  CreateAclTensor(outData, shape, &outAddr, ACL_INT8, &out);

  RunPowTensorTensor(stream, "TC-PTT-005", self, exp, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(exp, expAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-PTT-006: int16 x int16 -> int16 (opKey=6)
 */
void TestPowTensorTensor_Int16(aclrtStream stream) {
  TEST_CASE("TC-PTT-006: PowTensorTensor int16 x int16 (opKey=6)");
  std::vector<int64_t> shape = {8};
  std::vector<int16_t> selfData = {-3, -2, -1, 0, 1, 2, 3, 4};
  std::vector<int16_t> expData  = {2, 2, 2, 2, 3, 3, 3, 3};
  std::vector<int16_t> outData(8, 0);

  void *selfAddr = nullptr, *expAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *exp = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_INT16, &self);
  CreateAclTensor(expData, shape, &expAddr, ACL_INT16, &exp);
  CreateAclTensor(outData, shape, &outAddr, ACL_INT16, &out);

  RunPowTensorTensor(stream, "TC-PTT-006", self, exp, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(exp, expAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-PTT-007: int32 x int32 -> int32 (opKey=7)
 */
void TestPowTensorTensor_Int32(aclrtStream stream) {
  TEST_CASE("TC-PTT-007: PowTensorTensor int32 x int32 (opKey=7)");
  std::vector<int64_t> shape = {8};
  std::vector<int32_t> selfData = {-3, -2, -1, 0, 1, 2, 3, 4};
  std::vector<int32_t> expData  = {3, 3, 3, 3, 2, 2, 2, 2};
  std::vector<int32_t> outData(8, 0);

  void *selfAddr = nullptr, *expAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *exp = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_INT32, &self);
  CreateAclTensor(expData, shape, &expAddr, ACL_INT32, &exp);
  CreateAclTensor(outData, shape, &outAddr, ACL_INT32, &out);

  RunPowTensorTensor(stream, "TC-PTT-007", self, exp, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(exp, expAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-PTT-008: broadcast场景 (shape [4,1] x [1,4] -> [4,4])
 */
void TestPowTensorTensor_Broadcast(aclrtStream stream) {
  TEST_CASE("TC-PTT-008: PowTensorTensor broadcast [4,1] x [1,4] -> [4,4]");
  std::vector<int64_t> selfShape = {4, 1};
  std::vector<int64_t> expShape  = {1, 4};
  std::vector<int64_t> outShape  = {4, 4};
  std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expData  = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outData(16, 0.0f);

  void *selfAddr = nullptr, *expAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *exp = nullptr, *out = nullptr;
  CreateAclTensor(selfData, selfShape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(expData, expShape, &expAddr, ACL_FLOAT, &exp);
  CreateAclTensor(outData, outShape, &outAddr, ACL_FLOAT, &out);

  RunPowTensorTensor(stream, "TC-PTT-008", self, exp, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(exp, expAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-PTT-009: broadcast场景 (shape [8] x [1] -> [8])
 */
void TestPowTensorTensor_Broadcast_1D(aclrtStream stream) {
  TEST_CASE("TC-PTT-009: PowTensorTensor broadcast [8] x [1] -> [8]");
  std::vector<int64_t> selfShape = {8};
  std::vector<int64_t> expShape  = {1};
  std::vector<int64_t> outShape  = {8};
  std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> expData  = {2.0f};
  std::vector<float> outData(8, 0.0f);

  void *selfAddr = nullptr, *expAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *exp = nullptr, *out = nullptr;
  CreateAclTensor(selfData, selfShape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(expData, expShape, &expAddr, ACL_FLOAT, &exp);
  CreateAclTensor(outData, outShape, &outAddr, ACL_FLOAT, &out);

  RunPowTensorTensor(stream, "TC-PTT-009", self, exp, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(exp, expAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-PTT-010: Inplace PowTensorTensor (float32)
 */
void TestInplacePowTensorTensor_Float32(aclrtStream stream) {
  TEST_CASE("TC-PTT-010: InplacePowTensorTensor float32");
  std::vector<int64_t> shape = {8};
  std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> expData  = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

  void *selfAddr = nullptr, *expAddr = nullptr;
  aclTensor *self = nullptr, *exp = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(expData, shape, &expAddr, ACL_FLOAT, &exp);

  RunInplacePowTensorTensor(stream, "TC-PTT-010", self, exp);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(exp, expAddr);
}

/**
 * TC-PTT-011: Inplace PowTensorTensor (int32)
 */
void TestInplacePowTensorTensor_Int32(aclrtStream stream) {
  TEST_CASE("TC-PTT-011: InplacePowTensorTensor int32");
  std::vector<int64_t> shape = {8};
  std::vector<int32_t> selfData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int32_t> expData  = {2, 2, 2, 2, 2, 2, 2, 2};

  void *selfAddr = nullptr, *expAddr = nullptr;
  aclTensor *self = nullptr, *exp = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_INT32, &self);
  CreateAclTensor(expData, shape, &expAddr, ACL_INT32, &exp);

  RunInplacePowTensorTensor(stream, "TC-PTT-011", self, exp);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(exp, expAddr);
}

/**
 * TC-PTT-012: 空tensor (self为空) -> 提前返回
 */
void TestPowTensorTensor_EmptyTensor(aclrtStream stream) {
  TEST_CASE("TC-PTT-012: PowTensorTensor empty tensor (IsEmpty path)");
  std::vector<int64_t> shape = {0, 4};
  void *selfAddr = nullptr, *expAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *exp = nullptr, *out = nullptr;
  CreateEmptyAclTensor(shape, &selfAddr, ACL_FLOAT, &self);
  CreateEmptyAclTensor(shape, &expAddr, ACL_FLOAT, &exp);
  CreateEmptyAclTensor(shape, &outAddr, ACL_FLOAT, &out);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS && workspaceSize == 0) {
    LOG_PRINT("  Empty tensor: workspaceSize=0 as expected\n");
    TEST_PASS("TC-PTT-012");
  } else {
    LOG_PRINT("  Empty tensor: ret=%d, workspaceSize=%lu\n", ret, workspaceSize);
    TEST_FAIL("TC-PTT-012", ret);
  }

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(exp, expAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-PTT-013: 大shape float32 (触发多核tiling)
 */
void TestPowTensorTensor_Float32_LargeShape(aclrtStream stream) {
  TEST_CASE("TC-PTT-013: PowTensorTensor float32 large shape (multi-core tiling)");
  std::vector<int64_t> shape = {512, 512};
  int64_t n = 512 * 512;
  std::vector<float> selfData(n, 2.0f);
  std::vector<float> expData(n, 3.0f);
  std::vector<float> outData(n, 0.0f);

  void *selfAddr = nullptr, *expAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *exp = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(expData, shape, &expAddr, ACL_FLOAT, &exp);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);

  RunPowTensorTensor(stream, "TC-PTT-013", self, exp, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(exp, expAddr);
  FreeTensorAndMem(out, outAddr);
}

// ============================================================
// Exp2 测试
// ============================================================

/**
 * 通用Exp2执行函数
 */
int RunExp2(aclrtStream stream, const char* caseName,
               aclTensor* self, aclTensor* out, void* outAddr,
               const std::vector<double>& expected = {}, bool expectError = false) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  void* workspaceAddr = nullptr;

  auto ret = aclnnExp2GetWorkspaceSize(self, out, &workspaceSize, &executor);
  if (expectError) {
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("  [Expected error] aclnnExp2GetWorkspaceSize returned %d\n", ret);
      TEST_PASS(caseName);
      return 0;
    }
    LOG_PRINT("  [Unexpected success] Expected error but got ACL_SUCCESS\n");
    TEST_FAIL(caseName, -1);
    return -1;
  }
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclnnExp2GetWorkspaceSize failed. ERROR: %d\n", ret);
    TEST_FAIL(caseName, ret); return ret);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("  allocate workspace failed. ERROR: %d\n", ret);
      TEST_FAIL(caseName, ret); return ret);
  }

  ret = aclnnExp2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclnnExp2 failed. ERROR: %d\n", ret);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    TEST_FAIL(caseName, ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    TEST_FAIL(caseName, ret); return ret);

  if (workspaceAddr) aclrtFree(workspaceAddr);

  if (!expected.empty()) {
    if (strstr(caseName, "float32")) return VerifyResult<float>(out, outAddr, expected, caseName);
    if (strstr(caseName, "float16") || strstr(caseName, "fp16") || strstr(caseName, "bf16"))
        return VerifyResult<uint16_t>(out, outAddr, expected, caseName);
    if (strstr(caseName, "int32")) return VerifyResult<int32_t>(out, outAddr, expected, caseName);
    return VerifyResult<float>(out, outAddr, expected, caseName);
  }

  TEST_PASS(caseName);
  return 0;
}

/**
 * 通用InplaceExp2执行函数
 */
int RunInplaceExp2(aclrtStream stream, const char* caseName, aclTensor* selfRef) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  void* workspaceAddr = nullptr;

  auto ret = aclnnInplaceExp2GetWorkspaceSize(selfRef, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclnnInplaceExp2GetWorkspaceSize failed. ERROR: %d\n", ret);
    TEST_FAIL(caseName, ret); return ret);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("  allocate workspace failed. ERROR: %d\n", ret);
      TEST_FAIL(caseName, ret); return ret);
  }

  ret = aclnnInplaceExp2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclnnInplaceExp2 failed. ERROR: %d\n", ret);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    TEST_FAIL(caseName, ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("  aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    TEST_FAIL(caseName, ret); return ret);

  if (workspaceAddr) aclrtFree(workspaceAddr);
  TEST_PASS(caseName);
  return 0;
}

/**
 * TC-EXP2-001: float32 基础用例 (直接路径, 不cast)
 */
void TestExp2_Float32(aclrtStream stream) {
  TEST_CASE("TC-EXP2-001: Exp2 float32 (direct path, no cast)");
  std::vector<int64_t> shape = {8};
  std::vector<float> selfData = {0.0f, 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 0.5f, -0.5f};
  std::vector<float> outData(8, 0.0f);

  std::vector<double> expected(8);
  for (int i = 0; i < 8; i++) {
    expected[i] = std::pow(2.0, (double)selfData[i]);
  }

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);

  RunExp2(stream, "TC-EXP2-001_float32", self, out, outAddr, expected);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-EXP2-002: float16 用例 (直接路径, 不cast)
 */
void TestExp2_Float16(aclrtStream stream) {
  TEST_CASE("TC-EXP2-002: Exp2 float16 (direct path)");
  std::vector<int64_t> shape = {8};
  std::vector<uint16_t> selfData = {
    FloatToFp16(0.0f), FloatToFp16(1.0f), FloatToFp16(2.0f), FloatToFp16(3.0f),
    FloatToFp16(-1.0f), FloatToFp16(-2.0f), FloatToFp16(0.5f), FloatToFp16(-0.5f)
  };
  std::vector<uint16_t> outData(8, 0);

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT16, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT16, &out);

  RunExp2(stream, "TC-EXP2-002", self, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-EXP2-003: bfloat16 用例 (先cast到float路径)
 */
void TestExp2_BFloat16(aclrtStream stream) {
  TEST_CASE("TC-EXP2-003: Exp2 bfloat16 (cast to float path)");
  std::vector<int64_t> shape = {8};
  std::vector<uint16_t> selfData(8);
  float vals[] = {0.0f, 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 0.5f, -0.5f};
  for (int i = 0; i < 8; i++) {
    uint32_t x; memcpy(&x, &vals[i], 4);
    selfData[i] = (uint16_t)(x >> 16);
  }
  std::vector<float> outData(8, 0.0f);

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_BF16, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);

  RunExp2(stream, "TC-EXP2-003", self, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-EXP2-004: int32 用例 (先cast到float路径)
 */
void TestExp2_Int32(aclrtStream stream) {
  TEST_CASE("TC-EXP2-004: Exp2 int32 (cast to float path)");
  std::vector<int64_t> shape = {8};
  std::vector<int32_t> selfData = {0, 1, 2, 3, -1, -2, 4, 5};
  std::vector<float> outData(8, 0.0f);

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_INT32, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);

  RunExp2(stream, "TC-EXP2-004", self, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-EXP2-005: int8 用例 (先cast到float路径)
 */
void TestExp2_Int8(aclrtStream stream) {
  TEST_CASE("TC-EXP2-005: Exp2 int8 (cast to float path)");
  std::vector<int64_t> shape = {8};
  std::vector<int8_t> selfData = {0, 1, 2, 3, -1, -2, 4, 5};
  std::vector<float> outData(8, 0.0f);

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_INT8, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);

  RunExp2(stream, "TC-EXP2-005", self, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-EXP2-006: uint8 用例 (先cast到float路径)
 */
void TestExp2_Uint8(aclrtStream stream) {
  TEST_CASE("TC-EXP2-006: Exp2 uint8 (cast to float path)");
  std::vector<int64_t> shape = {8};
  std::vector<uint8_t> selfData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> outData(8, 0.0f);

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_UINT8, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);

  RunExp2(stream, "TC-EXP2-006", self, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-EXP2-007: int16 用例 (先cast到float路径)
 */
void TestExp2_Int16(aclrtStream stream) {
  TEST_CASE("TC-EXP2-007: Exp2 int16 (cast to float path)");
  std::vector<int64_t> shape = {8};
  std::vector<int16_t> selfData = {0, 1, 2, 3, -1, -2, 4, 5};
  std::vector<float> outData(8, 0.0f);

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_INT16, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);

  RunExp2(stream, "TC-EXP2-007", self, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-EXP2-008: bool 用例 (先cast到float路径)
 */
void TestExp2_Bool(aclrtStream stream) {
  TEST_CASE("TC-EXP2-008: Exp2 bool (cast to float path)");
  std::vector<int64_t> shape = {4};
  std::vector<uint8_t> selfData = {0, 1, 0, 1};  // bool用uint8存储
  std::vector<float> outData(4, 0.0f);

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_BOOL, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);

  RunExp2(stream, "TC-EXP2-008", self, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-EXP2-009: 空tensor -> 提前返回
 */
void TestExp2_EmptyTensor(aclrtStream stream) {
  TEST_CASE("TC-EXP2-009: Exp2 empty tensor (IsEmpty path)");
  std::vector<int64_t> shape = {0, 4};
  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateEmptyAclTensor(shape, &selfAddr, ACL_FLOAT, &self);
  CreateEmptyAclTensor(shape, &outAddr, ACL_FLOAT, &out);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnExp2GetWorkspaceSize(self, out, &workspaceSize, &executor);
  if (ret == ACL_SUCCESS && workspaceSize == 0) {
    LOG_PRINT("  Empty tensor: workspaceSize=0 as expected\n");
    TEST_PASS("TC-EXP2-009");
  } else {
    LOG_PRINT("  Empty tensor: ret=%d, workspaceSize=%lu\n", ret, workspaceSize);
    TEST_FAIL("TC-EXP2-009", ret);
  }

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
}

/**
 * TC-EXP2-010: Inplace Exp2 (float32)
 */
void TestInplaceExp2_Float32(aclrtStream stream) {
  TEST_CASE("TC-EXP2-010: InplaceExp2 float32");
  std::vector<int64_t> shape = {8};
  std::vector<float> selfData = {0.0f, 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 0.5f, -0.5f};

  void* selfAddr = nullptr;
  aclTensor* self = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);

  RunInplaceExp2(stream, "TC-EXP2-010", self);

  FreeTensorAndMem(self, selfAddr);
}

/**
 * TC-EXP2-011: Inplace Exp2 (float16)
 */
void TestInplaceExp2_Float16(aclrtStream stream) {
  TEST_CASE("TC-EXP2-011: InplaceExp2 float16");
  std::vector<int64_t> shape = {8};
  std::vector<uint16_t> selfData = {
    FloatToFp16(0.0f), FloatToFp16(1.0f), FloatToFp16(2.0f), FloatToFp16(3.0f),
    FloatToFp16(-1.0f), FloatToFp16(-2.0f), FloatToFp16(0.5f), FloatToFp16(-0.5f)
  };

  void* selfAddr = nullptr;
  aclTensor* self = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT16, &self);

  RunInplaceExp2(stream, "TC-EXP2-011", self);

  FreeTensorAndMem(self, selfAddr);
}

/**
 * TC-EXP2-012: Inplace Exp2 (bfloat16, 仅910B+支持)
 */
void TestInplaceExp2_BFloat16(aclrtStream stream) {
  TEST_CASE("TC-EXP2-012: InplaceExp2 bfloat16 (910B+ only)");
  std::vector<int64_t> shape = {8};
  std::vector<uint16_t> selfData(8);
  float vals[] = {0.0f, 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 0.5f, -0.5f};
  for (int i = 0; i < 8; i++) {
    uint32_t x; memcpy(&x, &vals[i], 4);
    selfData[i] = (uint16_t)(x >> 16);
  }

  void* selfAddr = nullptr;
  aclTensor* self = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_BF16, &self);

  RunInplaceExp2(stream, "TC-EXP2-012", self);

  FreeTensorAndMem(self, selfAddr);
}

/**
 * TC-EXP2-013: 多维shape float32 (4D)
 */
void TestExp2_Float32_4D(aclrtStream stream) {
  TEST_CASE("TC-EXP2-013: Exp2 float32 4D shape");
  std::vector<int64_t> shape = {2, 2, 2, 2};
  std::vector<float> selfData = {0.0f, 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 0.5f, -0.5f,
                                  0.0f, 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 0.5f, -0.5f};
  std::vector<float> outData(16, 0.0f);

  void *selfAddr = nullptr, *outAddr = nullptr;
  aclTensor *self = nullptr, *out = nullptr;
  CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT, &self);
  CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT, &out);

  RunExp2(stream, "TC-EXP2-013", self, out, outAddr);

  FreeTensorAndMem(self, selfAddr);
  FreeTensorAndMem(out, outAddr);
}

// ============================================================
// main函数：运行所有测试用例
// ============================================================
int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  printf("========================================\n");
  printf("  Pow算子端到端测试\n");
  printf("========================================\n");

  // ---- PowTensorScalar: 7种dtype tiling路径 ----
  TestPowTensorScalar_Float32_NormalExp(stream);    // TC-PTS-001 tilingKey=3001
  TestPowTensorScalar_Float16_NormalExp(stream);    // TC-PTS-002 tilingKey=1001
  TestPowTensorScalar_BFloat16_NormalExp(stream);   // TC-PTS-003 tilingKey=2001
  TestPowTensorScalar_Uint8_PosExp(stream);         // TC-PTS-004 tilingKey=4001
  TestPowTensorScalar_Int8_PosExp(stream);          // TC-PTS-005 tilingKey=5001
  TestPowTensorScalar_Int16_PosExp(stream);         // TC-PTS-006 tilingKey=6001
  TestPowTensorScalar_Int32_PosExp(stream);         // TC-PTS-007 tilingKey=7001

  // ---- PowTensorScalar: 特殊指数值路径 ----
  TestPowTensorScalar_Float32_Sqrt(stream);         // TC-PTS-008 Pows/sqrt
  TestPowTensorScalar_Float32_Square(stream);       // TC-PTS-009 Square
  TestPowTensorScalar_Float32_Cube(stream);         // TC-PTS-010 Pows/cube
  TestPowTensorScalar_Float32_NegSqrt(stream);      // TC-PTS-011 Pows/rsqrt
  TestPowTensorScalar_Float32_NegOne(stream);       // TC-PTS-012 Pows/reciprocal
  TestPowTensorScalar_Float32_NegSquare(stream);    // TC-PTS-013 Pows/-2
  TestPowTensorScalar_Float16_Square(stream);       // TC-PTS-014 Square fp16

  // ---- PowTensorScalar: 边界/错误路径 ----
  TestPowTensorScalar_Int32_NegExp_Error(stream);   // TC-PTS-015 整数负指数报错
  TestPowTensorScalar_EmptyTensor(stream);          // TC-PTS-016 空tensor
  TestInplacePowTensorScalar_Float32(stream);       // TC-PTS-017 Inplace float32
  TestInplacePowTensorScalar_Float16_Square(stream);// TC-PTS-018 Inplace fp16 Square
  TestPowTensorScalar_Float32_ZeroExp(stream);      // TC-PTS-019 exponent=0
  TestPowTensorScalar_Float32_LargeShape(stream);   // TC-PTS-020 大shape多核
  TestPowTensorScalar_Int32_ZeroExp(stream);        // TC-PTS-021 int32 exponent=0
  TestPowTensorScalar_Float32_4D(stream);           // TC-PTS-022 4D shape

  // ---- PowScalarTensor ----
  TestPowScalarTensor_ScalarOne_Fill(stream);       // TC-PST-001 fill(1)分支
  TestPowScalarTensor_Float32_Normal(stream);       // TC-PST-002 普通路径
  TestPowScalarTensor_Float16_Exp(stream);          // TC-PST-003 fp16 exponent
  TestPowScalarTensor_EmptyExponent(stream);        // TC-PST-004 空exponent

  // ---- PowTensorTensor: 7种dtype tiling路径 ----
  TestPowTensorTensor_Float32(stream);              // TC-PTT-001 opKey=3
  TestPowTensorTensor_Float16(stream);              // TC-PTT-002 opKey=1
  TestPowTensorTensor_BFloat16(stream);             // TC-PTT-003 opKey=2
  TestPowTensorTensor_Uint8(stream);                // TC-PTT-004 opKey=4
  TestPowTensorTensor_Int8(stream);                 // TC-PTT-005 opKey=5
  TestPowTensorTensor_Int16(stream);                // TC-PTT-006 opKey=6
  TestPowTensorTensor_Int32(stream);                // TC-PTT-007 opKey=7

  // ---- PowTensorTensor: broadcast/边界 ----
  TestPowTensorTensor_Broadcast(stream);            // TC-PTT-008 broadcast 2D
  TestPowTensorTensor_Broadcast_1D(stream);         // TC-PTT-009 broadcast 1D
  TestInplacePowTensorTensor_Float32(stream);       // TC-PTT-010 Inplace float32
  TestInplacePowTensorTensor_Int32(stream);         // TC-PTT-011 Inplace int32
  TestPowTensorTensor_EmptyTensor(stream);          // TC-PTT-012 空tensor
  TestPowTensorTensor_Float32_LargeShape(stream);   // TC-PTT-013 大shape多核

  // ---- Exp2 ----
  TestExp2_Float32(stream);                         // TC-EXP2-001 float32直接路径
  TestExp2_Float16(stream);                         // TC-EXP2-002 float16直接路径
  TestExp2_BFloat16(stream);                        // TC-EXP2-003 bf16 cast路径
  TestExp2_Int32(stream);                           // TC-EXP2-004 int32 cast路径
  TestExp2_Int8(stream);                            // TC-EXP2-005 int8 cast路径
  TestExp2_Uint8(stream);                           // TC-EXP2-006 uint8 cast路径
  TestExp2_Int16(stream);                           // TC-EXP2-007 int16 cast路径
  TestExp2_Bool(stream);                            // TC-EXP2-008 bool cast路径
  TestExp2_EmptyTensor(stream);                     // TC-EXP2-009 空tensor
  TestInplaceExp2_Float32(stream);                  // TC-EXP2-010 Inplace float32
  TestInplaceExp2_Float16(stream);                  // TC-EXP2-011 Inplace float16
  TestInplaceExp2_BFloat16(stream);                 // TC-EXP2-012 Inplace bf16
  TestExp2_Float32_4D(stream);                      // TC-EXP2-013 4D shape

  // ---- 补充覆盖率用例 ----

  // TC-ERR-001: PowTensorScalar bool×bool -> 报错 (aclnn_pow.cpp:239)
  {
    TEST_CASE("TC-ERR-001: PowTensorScalar bool x bool -> error");
    std::vector<int64_t> shape = {4};
    std::vector<uint8_t> selfData = {0, 1, 0, 1};
    std::vector<uint8_t> outData(4, 0);
    void *selfAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    CreateAclTensor(selfData, shape, &selfAddr, ACL_BOOL, &self);
    CreateAclTensor(outData, shape, &outAddr, ACL_BOOL, &out);
    uint8_t expVal = 1;
    aclScalar* exponent = aclCreateScalar(&expVal, ACL_BOOL);
    RunPowTensorScalar(stream, "TC-ERR-001", self, exponent, out, outAddr, {}, /*expectError=*/true);
    FreeTensorAndMem(self, selfAddr); FreeTensorAndMem(out, outAddr);
    aclDestroyScalar(exponent);
  }

  // TC-ERR-002: PowTensorTensor bool×bool -> 报错 (aclnn_pow_tensor_tensor.cpp:75)
  {
    TEST_CASE("TC-ERR-002: PowTensorTensor bool x bool -> error");
    std::vector<int64_t> shape = {4};
    std::vector<uint8_t> selfData = {0, 1, 0, 1};
    std::vector<uint8_t> expData  = {1, 0, 1, 0};
    std::vector<uint8_t> outData(4, 0);
    void *selfAddr = nullptr, *expAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *exp = nullptr, *out = nullptr;
    CreateAclTensor(selfData, shape, &selfAddr, ACL_BOOL, &self);
    CreateAclTensor(expData,  shape, &expAddr,  ACL_BOOL, &exp);
    CreateAclTensor(outData,  shape, &outAddr,  ACL_BOOL, &out);
    RunPowTensorTensor(stream, "TC-ERR-002", self, exp, out, outAddr, {}, /*expectError=*/true);
    FreeTensorAndMem(self, selfAddr); FreeTensorAndMem(exp, expAddr); FreeTensorAndMem(out, outAddr);
  }

  // TC-ERR-003: PowTensorTensor out shape不匹配 -> 报错 (aclnn_pow_tensor_tensor.cpp:106)
  {
    TEST_CASE("TC-ERR-003: PowTensorTensor out shape mismatch -> error");
    std::vector<int64_t> selfShape = {4};
    std::vector<int64_t> expShape  = {4};
    std::vector<int64_t> outShape  = {8};  // 故意错误的shape
    std::vector<float> selfData(4, 2.0f);
    std::vector<float> expData(4, 2.0f);
    std::vector<float> outData(8, 0.0f);
    void *selfAddr = nullptr, *expAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *exp = nullptr, *out = nullptr;
    CreateAclTensor(selfData, selfShape, &selfAddr, ACL_FLOAT, &self);
    CreateAclTensor(expData,  expShape,  &expAddr,  ACL_FLOAT, &exp);
    CreateAclTensor(outData,  outShape,  &outAddr,  ACL_FLOAT, &out);
    RunPowTensorTensor(stream, "TC-ERR-003", self, exp, out, outAddr, {}, /*expectError=*/true);
    FreeTensorAndMem(self, selfAddr); FreeTensorAndMem(exp, expAddr); FreeTensorAndMem(out, outAddr);
  }

  // TC-ERR-004: fp16 tensor + exponent=0.5 -> 触发Pows路径fp16分支 (aclnn_pow.cpp:435~438)
  {
    TEST_CASE("TC-ERR-004: PowTensorScalar fp16 exponent=0.5 (Pows fp16 path)");
    std::vector<int64_t> shape = {8};
    std::vector<uint16_t> selfData = {
      FloatToFp16(1.0f), FloatToFp16(4.0f), FloatToFp16(9.0f), FloatToFp16(16.0f),
      FloatToFp16(25.0f), FloatToFp16(36.0f), FloatToFp16(49.0f), FloatToFp16(64.0f)
    };
    std::vector<uint16_t> outData(8, 0);
    float expVal = 0.5f;
    void *selfAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    CreateAclTensor(selfData, shape, &selfAddr, ACL_FLOAT16, &self);
    CreateAclTensor(outData,  shape, &outAddr,  ACL_FLOAT16, &out);
    aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);
    RunPowTensorScalar(stream, "TC-ERR-004", self, exponent, out, outAddr);
    FreeTensorAndMem(self, selfAddr); FreeTensorAndMem(out, outAddr);
    aclDestroyScalar(exponent);
  }

  // TC-ERR-005: int8 tensor + exponent=2.0 -> Square需要cast到int32 (aclnn_pow.cpp:442)
  {
    TEST_CASE("TC-ERR-005: PowTensorScalar int8 exponent=2.0 (Square cast to int32)");
    std::vector<int64_t> shape = {8};
    std::vector<int8_t> selfData = {-3, -2, -1, 0, 1, 2, 3, 4};
    std::vector<int8_t> outData(8, 0);
    float expVal = 2.0f;
    void *selfAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    CreateAclTensor(selfData, shape, &selfAddr, ACL_INT8, &self);
    CreateAclTensor(outData,  shape, &outAddr,  ACL_INT8, &out);
    aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);
    RunPowTensorScalar(stream, "TC-ERR-005", self, exponent, out, outAddr);
    FreeTensorAndMem(self, selfAddr); FreeTensorAndMem(out, outAddr);
    aclDestroyScalar(exponent);
  }

  // TC-ERR-006: uint8 tensor + exponent=2.0 -> Square需要cast到int32
  {
    TEST_CASE("TC-ERR-006: PowTensorScalar uint8 exponent=2.0 (Square cast to int32)");
    std::vector<int64_t> shape = {8};
    std::vector<uint8_t> selfData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<uint8_t> outData(8, 0);
    float expVal = 2.0f;
    void *selfAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    CreateAclTensor(selfData, shape, &selfAddr, ACL_UINT8, &self);
    CreateAclTensor(outData,  shape, &outAddr,  ACL_UINT8, &out);
    aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);
    RunPowTensorScalar(stream, "TC-ERR-006", self, exponent, out, outAddr);
    FreeTensorAndMem(self, selfAddr); FreeTensorAndMem(out, outAddr);
    aclDestroyScalar(exponent);
  }

  // TC-ERR-007: int16 tensor + exponent=2.0 -> Square需要cast到int32
  {
    TEST_CASE("TC-ERR-007: PowTensorScalar int16 exponent=2.0 (Square cast to int32)");
    std::vector<int64_t> shape = {8};
    std::vector<int16_t> selfData = {-3, -2, -1, 0, 1, 2, 3, 4};
    std::vector<int16_t> outData(8, 0);
    float expVal = 2.0f;
    void *selfAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    CreateAclTensor(selfData, shape, &selfAddr, ACL_INT16, &self);
    CreateAclTensor(outData,  shape, &outAddr,  ACL_INT16, &out);
    aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);
    RunPowTensorScalar(stream, "TC-ERR-007", self, exponent, out, outAddr);
    FreeTensorAndMem(self, selfAddr); FreeTensorAndMem(out, outAddr);
    aclDestroyScalar(exponent);
  }

  // TC-ERR-008: InplaceExp2 空tensor -> 提前返回 (aclnn_exp2.cpp:218)
  {
    TEST_CASE("TC-ERR-008: InplaceExp2 empty tensor (IsEmpty path)");
    std::vector<int64_t> shape = {0};
    void* selfAddr = nullptr;
    aclTensor* self = nullptr;
    CreateEmptyAclTensor(shape, &selfAddr, ACL_FLOAT, &self);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceExp2GetWorkspaceSize(self, &workspaceSize, &executor);
    if (ret == ACL_SUCCESS && workspaceSize == 0) {
      LOG_PRINT("  InplaceExp2 empty: workspaceSize=0 as expected\n");
      TEST_PASS("TC-ERR-008");
    } else {
      LOG_PRINT("  InplaceExp2 empty: ret=%d, workspaceSize=%lu\n", ret, workspaceSize);
      TEST_FAIL("TC-ERR-008", ret);
    }
    FreeTensorAndMem(self, selfAddr);
  }

  // TC-ERR-009: bf16 tensor + exponent=0.5 -> 触发Pows路径bf16分支
  {
    TEST_CASE("TC-ERR-009: PowTensorScalar bf16 exponent=0.5 (Pows bf16 path)");
    std::vector<int64_t> shape = {8};
    std::vector<uint16_t> selfData(8);
    float vals[] = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f};
    for (int i = 0; i < 8; i++) {
      uint32_t x; memcpy(&x, &vals[i], 4);
      selfData[i] = (uint16_t)(x >> 16);
    }
    std::vector<uint16_t> outData(8, 0);
    float expVal = 0.5f;
    void *selfAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    CreateAclTensor(selfData, shape, &selfAddr, ACL_BF16, &self);
    CreateAclTensor(outData,  shape, &outAddr,  ACL_BF16, &out);
    aclScalar* exponent = aclCreateScalar(&expVal, ACL_FLOAT);
    RunPowTensorScalar(stream, "TC-ERR-009", self, exponent, out, outAddr);
    FreeTensorAndMem(self, selfAddr); FreeTensorAndMem(out, outAddr);
    aclDestroyScalar(exponent);
  }

  // TC-ERR-010: PowTensorScalar int64 tensor + 正整数指数 -> CheckNotOverflow INT64 case (aclnn_pow.cpp:310)
  {
    TEST_CASE("TC-ERR-010: PowTensorScalar int64 + positive exponent (CheckNotOverflow INT64)");
    std::vector<int64_t> shape = {8};
    std::vector<int64_t> selfData = {-3, -2, -1, 0, 1, 2, 3, 4};
    std::vector<int64_t> outData(8, 0);
    int64_t expVal = 3;
    void *selfAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    CreateAclTensor(selfData, shape, &selfAddr, ACL_INT64, &self);
    CreateAclTensor(outData,  shape, &outAddr,  ACL_INT64, &out);
    aclScalar* exponent = aclCreateScalar(&expVal, ACL_INT64);
    RunPowTensorScalar(stream, "TC-ERR-010", self, exponent, out, outAddr);
    FreeTensorAndMem(self, selfAddr); FreeTensorAndMem(out, outAddr);
    aclDestroyScalar(exponent);
  }

  // TC-ERR-011: PowScalarTensor scalar=1.0 + int32 exponent -> fill(1)分支完整路径 (aclnn_pow.cpp:576~577)
  {
    TEST_CASE("TC-ERR-011: PowScalarTensor scalar=1.0 int32 exponent (fill(1) full path)");
    std::vector<int64_t> shape = {16};
    std::vector<int32_t> expData(16, 3);
    std::vector<float> outData(16, 0.0f);
    float selfVal = 1.0f;
    void *expAddr = nullptr, *outAddr = nullptr;
    aclTensor *exponent = nullptr, *out = nullptr;
    CreateAclTensor(expData, shape, &expAddr, ACL_INT32, &exponent);
    CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT,  &out);
    aclScalar* self = aclCreateScalar(&selfVal, ACL_FLOAT);
    RunPowScalarTensor(stream, "TC-ERR-011", self, exponent, out, outAddr);
    FreeTensorAndMem(exponent, expAddr); FreeTensorAndMem(out, outAddr);
    aclDestroyScalar(self);
  }

  // TC-ERR-012: PowScalarTensor scalar=2.0 + int32 exponent -> 普通计算路径ViewCopy (aclnn_pow.cpp:510~511)
  {
    TEST_CASE("TC-ERR-012: PowScalarTensor scalar=2.0 int32 exponent (compute path ViewCopy)");
    std::vector<int64_t> shape = {16};
    std::vector<int32_t> expData = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    std::vector<float> outData(16, 0.0f);
    float selfVal = 2.0f;
    void *expAddr = nullptr, *outAddr = nullptr;
    aclTensor *exponent = nullptr, *out = nullptr;
    CreateAclTensor(expData, shape, &expAddr, ACL_INT32, &exponent);
    CreateAclTensor(outData, shape, &outAddr, ACL_FLOAT,  &out);
    aclScalar* self = aclCreateScalar(&selfVal, ACL_FLOAT);
    RunPowScalarTensor(stream, "TC-ERR-012", self, exponent, out, outAddr);
    FreeTensorAndMem(exponent, expAddr); FreeTensorAndMem(out, outAddr);
    aclDestroyScalar(self);
  }

  // TC-ERR-013: PowTensorTensor promote失败 -> 报错 (aclnn_pow_tensor_tensor.cpp:88)
  // bool tensor + float tensor -> promote应该成功，用int64+complex触发失败
  // 实际上用不支持的dtype组合触发CheckDtypeValid报错
  {
    TEST_CASE("TC-ERR-013: PowTensorScalar int32 neg exponent (int8 base, error path)");
    std::vector<int64_t> shape = {4};
    std::vector<int8_t> selfData = {1, 2, 3, 4};
    std::vector<int8_t> outData(4, 0);
    int64_t expVal = -2;  // 整数dtype + 负指数 -> 报错
    void *selfAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    CreateAclTensor(selfData, shape, &selfAddr, ACL_INT8, &self);
    CreateAclTensor(outData,  shape, &outAddr,  ACL_INT8, &out);
    aclScalar* exponent = aclCreateScalar(&expVal, ACL_INT64);
    RunPowTensorScalar(stream, "TC-ERR-013", self, exponent, out, outAddr, {}, /*expectError=*/true);
    FreeTensorAndMem(self, selfAddr); FreeTensorAndMem(out, outAddr);
    aclDestroyScalar(exponent);
  }

  // TC-ERR-014: PowTensorScalar int16 + 负指数 -> 报错
  {
    TEST_CASE("TC-ERR-014: PowTensorScalar int16 negative exponent -> error");
    std::vector<int64_t> shape = {4};
    std::vector<int16_t> selfData = {1, 2, 3, 4};
    std::vector<int16_t> outData(4, 0);
    int64_t expVal = -1;
    void *selfAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    CreateAclTensor(selfData, shape, &selfAddr, ACL_INT16, &self);
    CreateAclTensor(outData,  shape, &outAddr,  ACL_INT16, &out);
    aclScalar* exponent = aclCreateScalar(&expVal, ACL_INT64);
    RunPowTensorScalar(stream, "TC-ERR-014", self, exponent, out, outAddr, {}, /*expectError=*/true);
    FreeTensorAndMem(self, selfAddr); FreeTensorAndMem(out, outAddr);
    aclDestroyScalar(exponent);
  }

  // TC-ERR-015: PowTensorScalar uint8 + 负指数 -> 报错
  {
    TEST_CASE("TC-ERR-015: PowTensorScalar uint8 negative exponent -> error");
    std::vector<int64_t> shape = {4};
    std::vector<uint8_t> selfData = {1, 2, 3, 4};
    std::vector<uint8_t> outData(4, 0);
    int64_t expVal = -1;
    void *selfAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *out = nullptr;
    CreateAclTensor(selfData, shape, &selfAddr, ACL_UINT8, &self);
    CreateAclTensor(outData,  shape, &outAddr,  ACL_UINT8, &out);
    aclScalar* exponent = aclCreateScalar(&expVal, ACL_INT64);
    RunPowTensorScalar(stream, "TC-ERR-015", self, exponent, out, outAddr, {}, /*expectError=*/true);
    FreeTensorAndMem(self, selfAddr); FreeTensorAndMem(out, outAddr);
    aclDestroyScalar(exponent);
  }

  // ---- 汇总 ----
  printf("\n========================================\n");
  printf("  测试结果: PASS=%d, FAIL=%d, TOTAL=%d\n",
         g_passCount, g_failCount, g_passCount + g_failCount);
  printf("========================================\n");

  aclrtSynchronizeStream(stream);
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return g_failCount > 0 ? 1 : 0;
}
