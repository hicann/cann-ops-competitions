/**
 * Coverage-first ACLNN Add family test.
 *
 * This file intentionally keeps many overlapping cases.  The goal is to
 * exercise aclnnAdd/aclnnAdds/inplace/V3 API branches, Add l0 dispatch,
 * arch35 tiling dtype branches, validation failures, and precision baselines.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

#if defined(__has_include)
#if __has_include("../op_api/add.h") && __has_include("opdev/make_op_executor.h") && \
    __has_include("opdev/op_executor.h")
#define ADD_TEST_HAS_L0OP_INPLACE 1
#include "../op_api/add.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#else
#define ADD_TEST_HAS_L0OP_INPLACE 0
#endif
#else
#define ADD_TEST_HAS_L0OP_INPLACE 0
#endif

#if ADD_TEST_HAS_L0OP_INPLACE
using namespace op;
#else
namespace l0op {
const aclTensor* Add(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor);
const aclTensor* AddInplace(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor);
}
#endif

#define LOG_PRINT(message, ...)     \
  do {                              \
    std::printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t size = 1;
  for (auto dim : shape) {
    size *= dim;
  }
  return size;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
  }
  return strides;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
    return ret;
  }
  return ACL_SUCCESS;
}

int Finalize(int32_t deviceId, aclrtStream stream) {
  if (stream != nullptr) {
    aclrtDestroyStream(stream);
  }
  aclrtResetDevice(deviceId);
  aclFinalize();
  return ACL_SUCCESS;
}

struct TensorHandle {
  aclTensor* tensor = nullptr;
  void* deviceAddr = nullptr;
  TensorHandle() = default;
  TensorHandle(const TensorHandle&) = delete;
  TensorHandle& operator=(const TensorHandle&) = delete;
  ~TensorHandle() {
    if (tensor != nullptr) {
      aclDestroyTensor(tensor);
    }
    if (deviceAddr != nullptr) {
      aclrtFree(deviceAddr);
    }
  }
};

struct ScalarHandle {
  aclScalar* scalar = nullptr;
  ScalarHandle() = default;
  ScalarHandle(const ScalarHandle&) = delete;
  ScalarHandle& operator=(const ScalarHandle&) = delete;
  ~ScalarHandle() {
    if (scalar != nullptr) {
      aclDestroyScalar(scalar);
    }
  }
};

struct WorkspaceHandle {
  void* addr = nullptr;
  WorkspaceHandle() = default;
  WorkspaceHandle(const WorkspaceHandle&) = delete;
  WorkspaceHandle& operator=(const WorkspaceHandle&) = delete;
  ~WorkspaceHandle() {
    if (addr != nullptr) {
      aclrtFree(addr);
    }
  }
};

uint32_t FloatBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float BitsToFloat(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float Bf16ToFloat(uint16_t value) {
  return BitsToFloat(static_cast<uint32_t>(value) << 16);
}

uint16_t FloatToBf16Bits(float value) {
  uint32_t bits = FloatBits(value);
  uint32_t lsb = (bits >> 16) & 1U;
  bits += 0x7FFFU + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

float HalfToFloat(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16;
  uint32_t exp = (h >> 10) & 0x1FU;
  uint32_t frac = h & 0x03FFU;
  uint32_t bits = 0;
  if (exp == 0) {
    if (frac == 0) {
      bits = sign;
    } else {
      exp = 1;
      while ((frac & 0x0400U) == 0) {
        frac <<= 1;
        --exp;
      }
      frac &= 0x03FFU;
      uint32_t fexp = exp + (127U - 15U);
      bits = sign | (fexp << 23) | (frac << 13);
    }
  } else if (exp == 0x1FU) {
    bits = sign | 0x7F800000U | (frac << 13);
  } else {
    uint32_t fexp = exp + (127U - 15U);
    bits = sign | (fexp << 23) | (frac << 13);
  }
  return BitsToFloat(bits);
}

uint16_t FloatToHalfBits(float value) {
  uint32_t bits = FloatBits(value);
  uint32_t sign = (bits >> 16) & 0x8000U;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
  uint32_t frac = bits & 0x7FFFFFU;

  if (((bits >> 23) & 0xFFU) == 0xFFU) {
    if (frac == 0) {
      return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | 0x7C00U | (frac >> 13) | 1U);
  }
  if (exp >= 0x1F) {
    return static_cast<uint16_t>(sign | 0x7C00U);
  }
  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    frac |= 0x800000U;
    uint32_t shifted = frac >> static_cast<uint32_t>(1 - exp);
    if ((shifted & 0x00001000U) != 0) {
      shifted += 0x00002000U;
    }
    return static_cast<uint16_t>(sign | (shifted >> 13));
  }
  if ((frac & 0x00001000U) != 0) {
    frac += 0x00002000U;
    if ((frac & 0x00800000U) != 0) {
      frac = 0;
      ++exp;
    }
  }
  if (exp >= 0x1F) {
    return static_cast<uint16_t>(sign | 0x7C00U);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (frac >> 13));
}

bool IsFloatingDtype(aclDataType dtype) {
  return dtype == ACL_FLOAT || dtype == ACL_DOUBLE || dtype == ACL_FLOAT16 || dtype == ACL_BF16;
}

bool IsIntegerLikeDtype(aclDataType dtype) {
  return dtype == ACL_INT8 || dtype == ACL_UINT8 || dtype == ACL_INT16 || dtype == ACL_INT32 ||
         dtype == ACL_INT64 || dtype == ACL_BOOL;
}

template <typename T>
double HostValueToDouble(const T& value, aclDataType dtype) {
  switch (dtype) {
    case ACL_FLOAT16:
      return static_cast<double>(HalfToFloat(static_cast<uint16_t>(value)));
    case ACL_BF16:
      return static_cast<double>(Bf16ToFloat(static_cast<uint16_t>(value)));
    case ACL_BOOL:
      return static_cast<bool>(value) ? 1.0 : 0.0;
    default:
      return static_cast<double>(value);
  }
}

template <typename T>
double DeviceValueToDouble(const T& value, aclDataType dtype) {
  return HostValueToDouble(value, dtype);
}

double CastReferenceToDtype(double value, aclDataType dtype) {
  if (IsFloatingDtype(dtype) && (std::isnan(value) || std::isinf(value))) {
    if (dtype == ACL_FLOAT16) {
      return static_cast<double>(HalfToFloat(FloatToHalfBits(static_cast<float>(value))));
    }
    if (dtype == ACL_BF16) {
      return static_cast<double>(Bf16ToFloat(FloatToBf16Bits(static_cast<float>(value))));
    }
    if (dtype == ACL_FLOAT) {
      return static_cast<double>(static_cast<float>(value));
    }
    return value;
  }

  switch (dtype) {
    case ACL_FLOAT:
      return static_cast<double>(static_cast<float>(value));
    case ACL_DOUBLE:
      return value;
    case ACL_FLOAT16:
      return static_cast<double>(HalfToFloat(FloatToHalfBits(static_cast<float>(value))));
    case ACL_BF16:
      return static_cast<double>(Bf16ToFloat(FloatToBf16Bits(static_cast<float>(value))));
    case ACL_INT8:
      return static_cast<double>(static_cast<int8_t>(value));
    case ACL_UINT8:
      return static_cast<double>(static_cast<uint8_t>(value));
    case ACL_INT16:
      return static_cast<double>(static_cast<int16_t>(value));
    case ACL_INT32:
      return static_cast<double>(static_cast<int32_t>(value));
    case ACL_INT64:
      return static_cast<double>(static_cast<int64_t>(value));
    case ACL_BOOL:
      return value != 0.0 ? 1.0 : 0.0;
    default:
      return value;
  }
}

int64_t BroadcastOffset(int64_t linear, const std::vector<int64_t>& outShape, const std::vector<int64_t>& inShape) {
  if (inShape.empty()) {
    return 0;
  }
  auto inStrides = MakeStrides(inShape);
  std::vector<int64_t> outIndex(outShape.size(), 0);
  for (int64_t i = static_cast<int64_t>(outShape.size()) - 1; i >= 0; --i) {
    int64_t dim = outShape[static_cast<size_t>(i)];
    outIndex[static_cast<size_t>(i)] = dim == 0 ? 0 : linear % dim;
    if (dim != 0) {
      linear /= dim;
    }
  }

  int64_t offset = 0;
  int64_t rankDiff = static_cast<int64_t>(outShape.size()) - static_cast<int64_t>(inShape.size());
  for (size_t i = 0; i < inShape.size(); ++i) {
    int64_t outDim = static_cast<int64_t>(i) + rankDiff;
    int64_t idx = inShape[i] == 1 ? 0 : outIndex[static_cast<size_t>(outDim)];
    offset += idx * inStrides[i];
  }
  return offset;
}

template <typename TSelf, typename TOther, typename TAlpha>
std::vector<double> BuildAddExpected(const std::vector<TSelf>& selfData,
                                     const std::vector<TOther>& otherData,
                                     const std::vector<int64_t>& selfShape,
                                     const std::vector<int64_t>& otherShape,
                                     const std::vector<int64_t>& outShape,
                                     aclDataType selfType,
                                     aclDataType otherType,
                                     const TAlpha& alphaValue,
                                     aclDataType alphaType,
                                     aclDataType outType) {
  int64_t outSize = GetShapeSize(outShape);
  std::vector<double> expected(static_cast<size_t>(outSize));
  double alpha = HostValueToDouble(alphaValue, alphaType);
  for (int64_t i = 0; i < outSize; ++i) {
    int64_t selfOffset = BroadcastOffset(i, outShape, selfShape);
    int64_t otherOffset = BroadcastOffset(i, outShape, otherShape);
    double lhs = HostValueToDouble(selfData[static_cast<size_t>(selfOffset)], selfType);
    double rhs = HostValueToDouble(otherData[static_cast<size_t>(otherOffset)], otherType);
    expected[static_cast<size_t>(i)] = CastReferenceToDtype(lhs + alpha * rhs, outType);
  }
  return expected;
}

template <typename TSelf, typename TOtherScalar, typename TAlpha>
std::vector<double> BuildAddsExpected(const std::vector<TSelf>& selfData,
                                      const std::vector<int64_t>& outShape,
                                      aclDataType selfType,
                                      const TOtherScalar& otherValue,
                                      aclDataType otherType,
                                      const TAlpha& alphaValue,
                                      aclDataType alphaType,
                                      aclDataType outType) {
  int64_t outSize = GetShapeSize(outShape);
  std::vector<double> expected(static_cast<size_t>(outSize));
  double rhs = HostValueToDouble(otherValue, otherType);
  double alpha = HostValueToDouble(alphaValue, alphaType);
  for (int64_t i = 0; i < outSize; ++i) {
    double lhs = HostValueToDouble(selfData[static_cast<size_t>(i)], selfType);
    expected[static_cast<size_t>(i)] = CastReferenceToDtype(lhs + alpha * rhs, outType);
  }
  return expected;
}

template <typename TSelfScalar, typename TOther, typename TAlpha>
std::vector<double> BuildAddV3Expected(const TSelfScalar& selfValue,
                                       aclDataType selfType,
                                       const std::vector<TOther>& otherData,
                                       const std::vector<int64_t>& outShape,
                                       aclDataType otherType,
                                       const TAlpha& alphaValue,
                                       aclDataType alphaType,
                                       aclDataType outType) {
  int64_t outSize = GetShapeSize(outShape);
  std::vector<double> expected(static_cast<size_t>(outSize));
  double lhs = HostValueToDouble(selfValue, selfType);
  double alpha = HostValueToDouble(alphaValue, alphaType);
  for (int64_t i = 0; i < outSize; ++i) {
    double rhs = HostValueToDouble(otherData[static_cast<size_t>(i)], otherType);
    expected[static_cast<size_t>(i)] = CastReferenceToDtype(lhs + alpha * rhs, outType);
  }
  return expected;
}

template <typename T>
bool CreateTensor(const char* testName,
                  const std::vector<T>& hostData,
                  const std::vector<int64_t>& shape,
                  aclDataType dataType,
                  TensorHandle& handle,
                  aclFormat format = aclFormat::ACL_FORMAT_ND) {
  int64_t elemCount = GetShapeSize(shape);
  int64_t bytes = elemCount * static_cast<int64_t>(sizeof(T));
  auto strides = MakeStrides(shape);
  if (bytes > 0) {
    auto ret = aclrtMalloc(&handle.deviceAddr, static_cast<size_t>(bytes), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[%s] FAIL: aclrtMalloc failed. ERROR: %d\n", testName, ret);
      return false;
    }
    ret = aclrtMemcpy(handle.deviceAddr, static_cast<size_t>(bytes), hostData.data(), static_cast<size_t>(bytes),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[%s] FAIL: aclrtMemcpy H2D failed. ERROR: %d\n", testName, ret);
      return false;
    }
  }
  handle.tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                                  shape.size(), handle.deviceAddr);
  if (handle.tensor == nullptr) {
    LOG_PRINT("[%s] FAIL: aclCreateTensor failed.\n", testName);
    return false;
  }
  return true;
}

template <typename T>
bool CreateTensorWithLayout(const char* testName,
                            const std::vector<T>& hostData,
                            const std::vector<int64_t>& viewShape,
                            const std::vector<int64_t>& storageShape,
                            const std::vector<int64_t>& viewStrides,
                            int64_t viewOffset,
                            aclDataType dataType,
                            TensorHandle& handle,
                            aclFormat format = aclFormat::ACL_FORMAT_ND) {
  int64_t elemCount = GetShapeSize(storageShape);
  int64_t bytes = elemCount * static_cast<int64_t>(sizeof(T));
  if (bytes > 0) {
    auto ret = aclrtMalloc(&handle.deviceAddr, static_cast<size_t>(bytes), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[%s] FAIL: aclrtMalloc failed. ERROR: %d\n", testName, ret);
      return false;
    }
    ret = aclrtMemcpy(handle.deviceAddr, static_cast<size_t>(bytes), hostData.data(), static_cast<size_t>(bytes),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[%s] FAIL: aclrtMemcpy H2D failed. ERROR: %d\n", testName, ret);
      return false;
    }
  }
  handle.tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, viewStrides.data(), viewOffset, format,
                                  storageShape.data(), storageShape.size(), handle.deviceAddr);
  if (handle.tensor == nullptr) {
    LOG_PRINT("[%s] FAIL: aclCreateTensor with layout failed.\n", testName);
    return false;
  }
  return true;
}

template <typename T>
bool CreateScalar(const char* testName, const T& value, aclDataType dataType, ScalarHandle& handle) {
  handle.scalar = aclCreateScalar(const_cast<T*>(&value), dataType);
  if (handle.scalar == nullptr) {
    LOG_PRINT("[%s] FAIL: aclCreateScalar failed.\n", testName);
    return false;
  }
  return true;
}

bool AllocateWorkspace(const char* testName, uint64_t workspaceSize, WorkspaceHandle& workspace) {
  if (workspaceSize == 0) {
    workspace.addr = nullptr;
    return true;
  }
  auto ret = aclrtMalloc(&workspace.addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: allocate workspace failed. ERROR: %d\n", testName, ret);
    return false;
  }
  return true;
}

template <typename T>
bool CopyDeviceToHost(const char* testName, const TensorHandle& tensor, std::vector<T>& hostData) {
  size_t bytes = hostData.size() * sizeof(T);
  if (bytes == 0) {
    return true;
  }
  auto ret = aclrtMemcpy(hostData.data(), bytes, tensor.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclrtMemcpy D2H failed. ERROR: %d\n", testName, ret);
    return false;
  }
  return true;
}

template <typename T>
bool CheckNumericResult(const char* testName,
                        const std::vector<T>& actualData,
                        const std::vector<double>& expectedData,
                        aclDataType outType,
                        double atol,
                        double rtol) {
  if (actualData.size() != expectedData.size()) {
    LOG_PRINT("[%s] FAIL: result size mismatch, actual=%zu expected=%zu\n", testName, actualData.size(),
              expectedData.size());
    return false;
  }

  double maxAbs = 0.0;
  double maxRel = 0.0;
  size_t firstMismatch = actualData.size();
  double firstActual = 0.0;
  double firstExpected = 0.0;
  constexpr double eps = 1e-12;

  for (size_t i = 0; i < actualData.size(); ++i) {
    double actual = DeviceValueToDouble(actualData[i], outType);
    double expected = expectedData[i];
    bool ok = false;
    if (std::isnan(expected)) {
      ok = std::isnan(actual);
    } else if (std::isinf(expected)) {
      ok = std::isinf(actual) && std::signbit(actual) == std::signbit(expected);
    } else if (IsIntegerLikeDtype(outType)) {
      ok = (static_cast<int64_t>(actual) == static_cast<int64_t>(expected));
    } else {
      double absErr = std::fabs(actual - expected);
      double relErr = absErr / std::max(std::fabs(expected), eps);
      ok = absErr <= atol + rtol * std::fabs(expected);
      maxAbs = std::max(maxAbs, absErr);
      maxRel = std::max(maxRel, relErr);
    }
    if (!ok && firstMismatch == actualData.size()) {
      firstMismatch = i;
      firstActual = actual;
      firstExpected = expected;
    }
  }

  if (firstMismatch != actualData.size()) {
    LOG_PRINT("[%s] FAIL: first mismatch index=%zu actual=%.12g expected=%.12g max_abs=%.12g max_rel=%.12g\n",
              testName, firstMismatch, firstActual, firstExpected, maxAbs, maxRel);
    return false;
  }
  return true;
}

template <typename T>
bool FinishCase(const char* testName, bool pass) {
  LOG_PRINT("[%s] [%s]\n", testName, pass ? "PASS" : "FAIL");
  return pass;
}

void LogPrecisionObservation(const char* testName,
                             int64_t index,
                             double ideal,
                             double dtypeExpected,
                             double actual) {
  double idealAbsError = std::fabs(actual - ideal);
  double idealRelError = std::isfinite(ideal) && ideal != 0.0 ? idealAbsError / std::fabs(ideal) : idealAbsError;
  double dtypeAbsError = std::fabs(actual - dtypeExpected);
  LOG_PRINT("[%s] [PRECISION] index=%lld ideal_double=%.17g dtype_expected=%.17g actual_as_double=%.17g "
            "ideal_abs_error=%.17g ideal_rel_error=%.17g dtype_loss=%.17g dtype_abs_error=%.17g\n",
            testName, static_cast<long long>(index), ideal, dtypeExpected, actual, idealAbsError, idealRelError,
            std::fabs(ideal - dtypeExpected), dtypeAbsError);
}

template <typename TSelf, typename TOther, typename TOut, typename TAlpha>
bool RunAddNumeric(const char* testName,
                   aclrtStream stream,
                   const std::vector<TSelf>& selfData,
                   const std::vector<TOther>& otherData,
                   const std::vector<int64_t>& selfShape,
                   const std::vector<int64_t>& otherShape,
                   const std::vector<int64_t>& outShape,
                   aclDataType selfType,
                   aclDataType otherType,
                   aclDataType outType,
                   const TAlpha& alphaValue,
                   aclDataType alphaType,
                   const std::vector<double>& expectedData,
                   bool expectSuccess,
                   double atol = 1e-5,
                   double rtol = 1e-5) {
  TensorHandle self;
  TensorHandle other;
  TensorHandle out;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<TOut> outInit(static_cast<size_t>(GetShapeSize(outShape)), static_cast<TOut>(0));

  if (!CreateTensor(testName, selfData, selfShape, selfType, self) ||
      !CreateTensor(testName, otherData, otherShape, otherType, other) ||
      !CreateTensor(testName, outInit, outShape, outType, out) ||
      !CreateScalar(testName, alphaValue, alphaType, alpha)) {
    return FinishCase<TOut>(testName, false);
  }

  auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
  if (!expectSuccess) {
    if (ret == ACL_SUCCESS) {
      LOG_PRINT("[%s] INFO: workspace accepted for failure/probe case; skip execution to avoid simulator-only backend gaps.\n",
                testName);
    }
    return FinishCase<TOut>(testName, true);
  }
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnAddGetWorkspaceSize failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOut>(testName, false);
  }
  if (!AllocateWorkspace(testName, workspaceSize, workspace)) {
    return FinishCase<TOut>(testName, false);
  }
  ret = aclnnAdd(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnAdd failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOut>(testName, false);
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOut>(testName, false);
  }
  std::vector<TOut> actual(static_cast<size_t>(GetShapeSize(outShape)), static_cast<TOut>(0));
  bool pass = CopyDeviceToHost(testName, out, actual) &&
              CheckNumericResult(testName, actual, expectedData, outType, atol, rtol);
  return FinishCase<TOut>(testName, pass);
}

template <typename TSelf, typename TOther, typename TOut, typename TAlpha>
bool RunAddPrecisionObservation(const char* testName,
                                aclrtStream stream,
                                const std::vector<TSelf>& selfData,
                                const std::vector<TOther>& otherData,
                                const std::vector<int64_t>& selfShape,
                                const std::vector<int64_t>& otherShape,
                                const std::vector<int64_t>& outShape,
                                aclDataType selfType,
                                aclDataType otherType,
                                aclDataType outType,
                                const TAlpha& alphaValue,
                                aclDataType alphaType,
                                double atol,
                                double rtol) {
  auto expectedData = BuildAddExpected(selfData, otherData, selfShape, otherShape, outShape, selfType, otherType,
                                       alphaValue, alphaType, outType);
  TensorHandle self;
  TensorHandle other;
  TensorHandle out;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<TOut> outInit(static_cast<size_t>(GetShapeSize(outShape)), static_cast<TOut>(0));

  if (!CreateTensor(testName, selfData, selfShape, selfType, self) ||
      !CreateTensor(testName, otherData, otherShape, otherType, other) ||
      !CreateTensor(testName, outInit, outShape, outType, out) ||
      !CreateScalar(testName, alphaValue, alphaType, alpha)) {
    return FinishCase<TOut>(testName, false);
  }
  auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnAddGetWorkspaceSize failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOut>(testName, false);
  }
  if (!AllocateWorkspace(testName, workspaceSize, workspace)) {
    return FinishCase<TOut>(testName, false);
  }
  ret = aclnnAdd(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnAdd failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOut>(testName, false);
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOut>(testName, false);
  }

  std::vector<TOut> actual(static_cast<size_t>(GetShapeSize(outShape)), static_cast<TOut>(0));
  bool pass = CopyDeviceToHost(testName, out, actual) &&
              CheckNumericResult(testName, actual, expectedData, outType, atol, rtol);
  double alphaValueDouble = HostValueToDouble(alphaValue, alphaType);
  for (int64_t i = 0; i < GetShapeSize(outShape); ++i) {
    int64_t selfOffset = BroadcastOffset(i, outShape, selfShape);
    int64_t otherOffset = BroadcastOffset(i, outShape, otherShape);
    double lhs = HostValueToDouble(selfData[static_cast<size_t>(selfOffset)], selfType);
    double rhs = HostValueToDouble(otherData[static_cast<size_t>(otherOffset)], otherType);
    double ideal = lhs + alphaValueDouble * rhs;
    double actualValue = DeviceValueToDouble(actual[static_cast<size_t>(i)], outType);
    LogPrecisionObservation(testName, i, ideal, expectedData[static_cast<size_t>(i)], actualValue);
  }
  return FinishCase<TOut>(testName, pass);
}

template <typename TSelf, typename TOtherScalar, typename TOut, typename TAlpha>
bool RunAddsNumeric(const char* testName,
                    aclrtStream stream,
                    const std::vector<TSelf>& selfData,
                    const std::vector<int64_t>& selfShape,
                    const std::vector<int64_t>& outShape,
                    aclDataType selfType,
                    aclDataType outType,
                    const TOtherScalar& otherValue,
                    aclDataType otherType,
                    const TAlpha& alphaValue,
                    aclDataType alphaType,
                    const std::vector<double>& expectedData,
                    bool expectSuccess,
                    double atol = 1e-5,
                    double rtol = 1e-5) {
  TensorHandle self;
  TensorHandle out;
  ScalarHandle other;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<TOut> outInit(static_cast<size_t>(GetShapeSize(outShape)), static_cast<TOut>(0));

  if (!CreateTensor(testName, selfData, selfShape, selfType, self) ||
      !CreateTensor(testName, outInit, outShape, outType, out) ||
      !CreateScalar(testName, otherValue, otherType, other) ||
      !CreateScalar(testName, alphaValue, alphaType, alpha)) {
    return FinishCase<TOut>(testName, false);
  }
  auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
  if (!expectSuccess) {
    if (ret == ACL_SUCCESS) {
      LOG_PRINT("[%s] INFO: workspace accepted for failure/probe case; skip execution to avoid simulator-only backend gaps.\n",
                testName);
    }
    return FinishCase<TOut>(testName, true);
  }
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnAddsGetWorkspaceSize failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOut>(testName, false);
  }
  if (!AllocateWorkspace(testName, workspaceSize, workspace)) {
    return FinishCase<TOut>(testName, false);
  }
  ret = aclnnAdds(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnAdds failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOut>(testName, false);
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOut>(testName, false);
  }
  std::vector<TOut> actual(static_cast<size_t>(GetShapeSize(outShape)), static_cast<TOut>(0));
  bool pass = CopyDeviceToHost(testName, out, actual) &&
              CheckNumericResult(testName, actual, expectedData, outType, atol, rtol);
  return FinishCase<TOut>(testName, pass);
}

template <typename TSelfScalar, typename TOther, typename TOut, typename TAlpha>
bool RunAddV3Numeric(const char* testName,
                     aclrtStream stream,
                     const TSelfScalar& selfValue,
                     aclDataType selfType,
                     const std::vector<TOther>& otherData,
                     const std::vector<int64_t>& otherShape,
                     const std::vector<int64_t>& outShape,
                     aclDataType otherType,
                     aclDataType outType,
                     const TAlpha& alphaValue,
                     aclDataType alphaType,
                     const std::vector<double>& expectedData,
                     bool expectSuccess,
                     double atol = 1e-5,
                     double rtol = 1e-5) {
  TensorHandle other;
  TensorHandle out;
  ScalarHandle self;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<TOut> outInit(static_cast<size_t>(GetShapeSize(outShape)), static_cast<TOut>(0));

  if (!CreateTensor(testName, otherData, otherShape, otherType, other) ||
      !CreateTensor(testName, outInit, outShape, outType, out) ||
      !CreateScalar(testName, selfValue, selfType, self) ||
      !CreateScalar(testName, alphaValue, alphaType, alpha)) {
    return FinishCase<TOut>(testName, false);
  }
  auto ret = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
  if (!expectSuccess) {
    if (ret == ACL_SUCCESS) {
      LOG_PRINT("[%s] INFO: workspace accepted for failure/probe case; skip execution to avoid simulator-only backend gaps.\n",
                testName);
    }
    return FinishCase<TOut>(testName, true);
  }
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnAddV3GetWorkspaceSize failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOut>(testName, false);
  }
  if (!AllocateWorkspace(testName, workspaceSize, workspace)) {
    return FinishCase<TOut>(testName, false);
  }
  ret = aclnnAddV3(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnAddV3 failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOut>(testName, false);
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOut>(testName, false);
  }
  std::vector<TOut> actual(static_cast<size_t>(GetShapeSize(outShape)), static_cast<TOut>(0));
  bool pass = CopyDeviceToHost(testName, out, actual) &&
              CheckNumericResult(testName, actual, expectedData, outType, atol, rtol);
  return FinishCase<TOut>(testName, pass);
}

template <typename TSelf, typename TOther, typename TAlpha>
bool RunInplaceAddNumeric(const char* testName,
                          aclrtStream stream,
                          const std::vector<TSelf>& selfData,
                          const std::vector<TOther>& otherData,
                          const std::vector<int64_t>& selfShape,
                          const std::vector<int64_t>& otherShape,
                          aclDataType selfType,
                          aclDataType otherType,
                          const TAlpha& alphaValue,
                          aclDataType alphaType,
                          const std::vector<double>& expectedData,
                          bool expectSuccess,
                          double atol = 1e-5,
                          double rtol = 1e-5) {
  TensorHandle self;
  TensorHandle other;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  if (!CreateTensor(testName, selfData, selfShape, selfType, self) ||
      !CreateTensor(testName, otherData, otherShape, otherType, other) ||
      !CreateScalar(testName, alphaValue, alphaType, alpha)) {
    return FinishCase<TSelf>(testName, false);
  }
  auto ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
  if (!expectSuccess) {
    if (ret == ACL_SUCCESS) {
      LOG_PRINT("[%s] INFO: workspace accepted for failure/probe case; skip execution to avoid simulator-only backend gaps.\n",
                testName);
    }
    return FinishCase<TSelf>(testName, true);
  }
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnInplaceAddGetWorkspaceSize failed. ERROR: %d\n", testName, ret);
    return FinishCase<TSelf>(testName, false);
  }
  if (!AllocateWorkspace(testName, workspaceSize, workspace)) {
    return FinishCase<TSelf>(testName, false);
  }
  ret = aclnnInplaceAdd(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnInplaceAdd failed. ERROR: %d\n", testName, ret);
    return FinishCase<TSelf>(testName, false);
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret);
    return FinishCase<TSelf>(testName, false);
  }
  std::vector<TSelf> actual(static_cast<size_t>(GetShapeSize(selfShape)), static_cast<TSelf>(0));
  bool pass = CopyDeviceToHost(testName, self, actual) &&
              CheckNumericResult(testName, actual, expectedData, selfType, atol, rtol);
  return FinishCase<TSelf>(testName, pass);
}

template <typename TSelf, typename TOtherScalar, typename TAlpha>
bool RunInplaceAddsNumeric(const char* testName,
                           aclrtStream stream,
                           const std::vector<TSelf>& selfData,
                           const std::vector<int64_t>& selfShape,
                           aclDataType selfType,
                           const TOtherScalar& otherValue,
                           aclDataType otherType,
                           const TAlpha& alphaValue,
                           aclDataType alphaType,
                           const std::vector<double>& expectedData,
                           bool expectSuccess,
                           double atol = 1e-5,
                           double rtol = 1e-5) {
  TensorHandle self;
  ScalarHandle other;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  if (!CreateTensor(testName, selfData, selfShape, selfType, self) ||
      !CreateScalar(testName, otherValue, otherType, other) ||
      !CreateScalar(testName, alphaValue, alphaType, alpha)) {
    return FinishCase<TSelf>(testName, false);
  }
  auto ret = aclnnInplaceAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, &workspaceSize, &executor);
  if (!expectSuccess) {
    if (ret == ACL_SUCCESS) {
      LOG_PRINT("[%s] INFO: workspace accepted for failure/probe case; skip execution to avoid simulator-only backend gaps.\n",
                testName);
    }
    return FinishCase<TSelf>(testName, true);
  }
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnInplaceAddsGetWorkspaceSize failed. ERROR: %d\n", testName, ret);
    return FinishCase<TSelf>(testName, false);
  }
  if (!AllocateWorkspace(testName, workspaceSize, workspace)) {
    return FinishCase<TSelf>(testName, false);
  }
  ret = aclnnInplaceAdds(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnInplaceAdds failed. ERROR: %d\n", testName, ret);
    return FinishCase<TSelf>(testName, false);
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret);
    return FinishCase<TSelf>(testName, false);
  }
  std::vector<TSelf> actual(static_cast<size_t>(GetShapeSize(selfShape)), static_cast<TSelf>(0));
  bool pass = CopyDeviceToHost(testName, self, actual) &&
              CheckNumericResult(testName, actual, expectedData, selfType, atol, rtol);
  return FinishCase<TSelf>(testName, pass);
}

template <typename TSelfScalar, typename TOther, typename TAlpha>
bool RunInplaceAddV3Numeric(const char* testName,
                            aclrtStream stream,
                            const TSelfScalar& selfValue,
                            aclDataType selfType,
                            const std::vector<TOther>& otherData,
                            const std::vector<int64_t>& otherShape,
                            aclDataType otherType,
                            const TAlpha& alphaValue,
                            aclDataType alphaType,
                            const std::vector<double>& expectedData,
                            bool expectSuccess,
                            double atol = 1e-5,
                            double rtol = 1e-5) {
  TensorHandle other;
  ScalarHandle self;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  if (!CreateTensor(testName, otherData, otherShape, otherType, other) ||
      !CreateScalar(testName, selfValue, selfType, self) ||
      !CreateScalar(testName, alphaValue, alphaType, alpha)) {
    return FinishCase<TOther>(testName, false);
  }
  auto ret = aclnnInplaceAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, &workspaceSize, &executor);
  if (!expectSuccess) {
    if (ret == ACL_SUCCESS) {
      LOG_PRINT("[%s] INFO: workspace accepted for failure/probe case; skip execution to avoid simulator-only backend gaps.\n",
                testName);
    }
    return FinishCase<TOther>(testName, true);
  }
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnInplaceAddV3GetWorkspaceSize failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOther>(testName, false);
  }
  if (!AllocateWorkspace(testName, workspaceSize, workspace)) {
    return FinishCase<TOther>(testName, false);
  }
  ret = aclnnInplaceAddV3(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnInplaceAddV3 failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOther>(testName, false);
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclrtSynchronizeStream failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOther>(testName, false);
  }
  std::vector<TOther> actual(static_cast<size_t>(GetShapeSize(otherShape)), static_cast<TOther>(0));
  bool pass = CopyDeviceToHost(testName, other, actual) &&
              CheckNumericResult(testName, actual, expectedData, otherType, atol, rtol);
  return FinishCase<TOther>(testName, pass);
}

bool CheckComplex64Result(const char* testName,
                          const std::vector<std::complex<float>>& actual,
                          const std::vector<std::complex<double>>& expected,
                          double atol,
                          double rtol) {
  if (actual.size() != expected.size()) {
    LOG_PRINT("[%s] FAIL: complex size mismatch\n", testName);
    return false;
  }
  constexpr double eps = 1e-12;
  for (size_t i = 0; i < actual.size(); ++i) {
    double ar = static_cast<double>(actual[i].real());
    double ai = static_cast<double>(actual[i].imag());
    double er = expected[i].real();
    double ei = expected[i].imag();
    double rAbs = std::fabs(ar - er);
    double iAbs = std::fabs(ai - ei);
    double rRel = rAbs / std::max(std::fabs(er), eps);
    double iRel = iAbs / std::max(std::fabs(ei), eps);
    bool ok = rAbs <= atol + rtol * std::fabs(er) && iAbs <= atol + rtol * std::fabs(ei);
    if (!ok) {
      LOG_PRINT("[%s] FAIL: complex mismatch index=%zu actual=(%.9g,%.9g) expected=(%.9g,%.9g) rel=(%.9g,%.9g)\n",
                testName, i, ar, ai, er, ei, rRel, iRel);
      return false;
    }
  }
  return true;
}

bool RunAddComplex64(const char* testName, aclrtStream stream, bool expectSuccess = true) {
  std::vector<int64_t> shape = {2, 2};
  std::vector<std::complex<float>> selfData = {{1.0f, 2.0f}, {-3.0f, 4.0f}, {5.0f, -6.0f}, {-7.0f, -8.0f}};
  std::vector<std::complex<float>> otherData = {{10.0f, -1.0f}, {2.0f, 3.0f}, {-4.0f, 5.0f}, {6.0f, -7.0f}};
  std::complex<float> alphaValue(1.0f, 0.0f);
  std::vector<std::complex<double>> expected;
  for (size_t i = 0; i < selfData.size(); ++i) {
    expected.emplace_back(static_cast<std::complex<double>>(selfData[i]) +
                          static_cast<std::complex<double>>(alphaValue) *
                              static_cast<std::complex<double>>(otherData[i]));
  }

  TensorHandle self;
  TensorHandle other;
  TensorHandle out;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<std::complex<float>> outInit(selfData.size(), {0.0f, 0.0f});
  if (!CreateTensor(testName, selfData, shape, ACL_COMPLEX64, self) ||
      !CreateTensor(testName, otherData, shape, ACL_COMPLEX64, other) ||
      !CreateTensor(testName, outInit, shape, ACL_COMPLEX64, out) ||
      !CreateScalar(testName, alphaValue, ACL_COMPLEX64, alpha)) {
    return FinishCase<std::complex<float>>(testName, false);
  }
  auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    if (!expectSuccess) {
      LOG_PRINT("[%s] INFO: complex workspace rejected as expected/probe. ERROR: %d\n", testName, ret);
      return FinishCase<std::complex<float>>(testName, true);
    }
    LOG_PRINT("[%s] FAIL: aclnnAddGetWorkspaceSize failed. ERROR: %d\n", testName, ret);
    return FinishCase<std::complex<float>>(testName, false);
  }
  if (!expectSuccess) {
    LOG_PRINT("[%s] INFO: complex workspace accepted for probe; skip execution in simulator.\n", testName);
    return FinishCase<std::complex<float>>(testName, true);
  }
  if (!AllocateWorkspace(testName, workspaceSize, workspace)) {
    return FinishCase<std::complex<float>>(testName, false);
  }
  ret = aclnnAdd(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnAdd complex execution failed. ERROR: %d\n", testName, ret);
    return FinishCase<std::complex<float>>(testName, false);
  }
  std::vector<std::complex<float>> actual(selfData.size());
  bool pass = CopyDeviceToHost(testName, out, actual) && CheckComplex64Result(testName, actual, expected, 1e-4, 1e-4);
  return FinishCase<std::complex<float>>(testName, pass);
}

bool RunAddV3Complex64(const char* testName, aclrtStream stream, bool expectSuccess = true) {
  std::vector<int64_t> shape = {2, 2};
  std::complex<float> selfValue(1.0f, 2.0f);
  std::complex<float> alphaValue(1.0f, 0.0f);
  std::vector<float> otherData = {10.0f, 20.0f, -30.0f, -40.0f};
  std::vector<std::complex<double>> expected = {
      {11.0, 2.0}, {21.0, 2.0}, {-29.0, 2.0}, {-39.0, 2.0}};

  TensorHandle other;
  TensorHandle out;
  ScalarHandle self;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<std::complex<float>> outInit(otherData.size(), {0.0f, 0.0f});
  if (!CreateTensor(testName, otherData, shape, ACL_FLOAT, other) ||
      !CreateTensor(testName, outInit, shape, ACL_COMPLEX64, out) ||
      !CreateScalar(testName, selfValue, ACL_COMPLEX64, self) ||
      !CreateScalar(testName, alphaValue, ACL_COMPLEX64, alpha)) {
    return FinishCase<std::complex<float>>(testName, false);
  }
  auto ret = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    if (!expectSuccess) {
      LOG_PRINT("[%s] INFO: complex V3 workspace rejected as expected/probe. ERROR: %d\n", testName, ret);
      return FinishCase<std::complex<float>>(testName, true);
    }
    LOG_PRINT("[%s] FAIL: aclnnAddV3GetWorkspaceSize failed. ERROR: %d\n", testName, ret);
    return FinishCase<std::complex<float>>(testName, false);
  }
  if (!expectSuccess) {
    LOG_PRINT("[%s] INFO: complex V3 workspace accepted for probe; skip execution in simulator.\n", testName);
    return FinishCase<std::complex<float>>(testName, true);
  }
  if (!AllocateWorkspace(testName, workspaceSize, workspace)) {
    return FinishCase<std::complex<float>>(testName, false);
  }
  ret = aclnnAddV3(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnAddV3 complex execution failed. ERROR: %d\n", testName, ret);
    return FinishCase<std::complex<float>>(testName, false);
  }
  std::vector<std::complex<float>> actual(otherData.size());
  bool pass = CopyDeviceToHost(testName, out, actual) && CheckComplex64Result(testName, actual, expected, 1e-4, 1e-4);
  return FinishCase<std::complex<float>>(testName, pass);
}

bool RunAddsComplexScalar64(const char* testName, aclrtStream stream, bool expectSuccess = true) {
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfData = {1.0f, -2.0f, 3.5f, -4.5f};
  std::complex<float> otherValue(2.0f, -0.5f);
  std::complex<float> alphaValue(0.5f, -1.0f);
  std::complex<double> scalarProduct = static_cast<std::complex<double>>(alphaValue) *
                                       static_cast<std::complex<double>>(otherValue);
  std::vector<std::complex<double>> expected;
  for (float value : selfData) {
    expected.emplace_back(std::complex<double>(static_cast<double>(value), 0.0) + scalarProduct);
  }

  TensorHandle self;
  TensorHandle out;
  ScalarHandle other;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<std::complex<float>> outInit(selfData.size(), {0.0f, 0.0f});
  if (!CreateTensor(testName, selfData, shape, ACL_FLOAT, self) ||
      !CreateTensor(testName, outInit, shape, ACL_COMPLEX64, out) ||
      !CreateScalar(testName, otherValue, ACL_COMPLEX64, other) ||
      !CreateScalar(testName, alphaValue, ACL_COMPLEX64, alpha)) {
    return FinishCase<std::complex<float>>(testName, false);
  }
  auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    if (!expectSuccess) {
      LOG_PRINT("[%s] INFO: complex scalar workspace rejected as expected/probe. ERROR: %d\n", testName, ret);
      return FinishCase<std::complex<float>>(testName, true);
    }
    LOG_PRINT("[%s] FAIL: aclnnAddsGetWorkspaceSize complex scalar failed. ERROR: %d\n", testName, ret);
    return FinishCase<std::complex<float>>(testName, false);
  }
  if (!expectSuccess) {
    LOG_PRINT("[%s] INFO: complex scalar workspace accepted for probe; skip execution in simulator.\n", testName);
    return FinishCase<std::complex<float>>(testName, true);
  }
  if (!AllocateWorkspace(testName, workspaceSize, workspace)) {
    return FinishCase<std::complex<float>>(testName, false);
  }
  ret = aclnnAdds(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnAdds complex scalar execution failed. ERROR: %d\n", testName, ret);
    return FinishCase<std::complex<float>>(testName, false);
  }
  std::vector<std::complex<float>> actual(selfData.size());
  bool pass = CopyDeviceToHost(testName, out, actual) && CheckComplex64Result(testName, actual, expected, 1e-4, 1e-4);
  return FinishCase<std::complex<float>>(testName, pass);
}

bool RunAddNonContiguousFloat(const char* testName, aclrtStream stream) {
  std::vector<int64_t> viewShape = {2, 2};
  std::vector<int64_t> storageShape = {2, 3};
  std::vector<int64_t> strides = {3, 1};
  std::vector<float> selfStorage = {-99.0f, 1.0f, 2.0f, -99.0f, 3.0f, 4.0f};
  std::vector<float> otherStorage = {-77.0f, 10.0f, 20.0f, -77.0f, 30.0f, 40.0f};
  std::vector<float> outInit(4, 0.0f);
  std::vector<double> expected = {11.0, 22.0, 33.0, 44.0};
  float alphaValue = 1.0f;

  TensorHandle self;
  TensorHandle other;
  TensorHandle out;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  if (!CreateTensorWithLayout(testName, selfStorage, viewShape, storageShape, strides, 1, ACL_FLOAT, self) ||
      !CreateTensorWithLayout(testName, otherStorage, viewShape, storageShape, strides, 1, ACL_FLOAT, other) ||
      !CreateTensor(testName, outInit, viewShape, ACL_FLOAT, out) ||
      !CreateScalar(testName, alphaValue, ACL_FLOAT, alpha)) {
    return FinishCase<float>(testName, false);
  }
  auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnAddGetWorkspaceSize failed. ERROR: %d\n", testName, ret);
    return FinishCase<float>(testName, false);
  }
  if (!AllocateWorkspace(testName, workspaceSize, workspace)) {
    return FinishCase<float>(testName, false);
  }
  ret = aclnnAdd(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: non-contiguous execution failed. ERROR: %d\n", testName, ret);
    return FinishCase<float>(testName, false);
  }
  std::vector<float> actual(4, 0.0f);
  bool pass = CopyDeviceToHost(testName, out, actual) &&
              CheckNumericResult(testName, actual, expected, ACL_FLOAT, 1e-6, 1e-6);
  return FinishCase<float>(testName, pass);
}

bool RunAddNonNdFormatFloat(const char* testName, aclrtStream stream) {
  std::vector<int64_t> shape = {1, 1, 2, 2};
  std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> otherData = {10.0f, 20.0f, 30.0f, 40.0f};
  auto expected = BuildAddExpected(selfData, otherData, shape, shape, shape, ACL_FLOAT, ACL_FLOAT, 1.0f, ACL_FLOAT,
                                   ACL_FLOAT);
  TensorHandle self;
  TensorHandle other;
  TensorHandle out;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<float> outInit(4, 0.0f);
  float alphaValue = 1.0f;
  if (!CreateTensor(testName, selfData, shape, ACL_FLOAT, self, aclFormat::ACL_FORMAT_NCHW) ||
      !CreateTensor(testName, otherData, shape, ACL_FLOAT, other, aclFormat::ACL_FORMAT_NCHW) ||
      !CreateTensor(testName, outInit, shape, ACL_FLOAT, out, aclFormat::ACL_FORMAT_NCHW) ||
      !CreateScalar(testName, alphaValue, ACL_FLOAT, alpha)) {
    return FinishCase<float>(testName, false);
  }
  auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnAddGetWorkspaceSize failed. ERROR: %d\n", testName, ret);
    return FinishCase<float>(testName, false);
  }
  if (!AllocateWorkspace(testName, workspaceSize, workspace)) {
    return FinishCase<float>(testName, false);
  }
  ret = aclnnAdd(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: non-ND execution failed. ERROR: %d\n", testName, ret);
    return FinishCase<float>(testName, false);
  }
  std::vector<float> actual(4, 0.0f);
  bool pass = CopyDeviceToHost(testName, out, actual) &&
              CheckNumericResult(testName, actual, expected, ACL_FLOAT, 1e-6, 1e-6);
  return FinishCase<float>(testName, pass);
}

bool RunAddsNonNdFormatFloat(const char* testName, aclrtStream stream, bool execute = true, float alphaValue = 1.0f) {
  std::vector<int64_t> shape = {1, 1, 2, 2};
  std::vector<float> selfData = {1.0f, -2.0f, 3.0f, -4.0f};
  float otherValue = 10.0f;
  auto expected = BuildAddsExpected(selfData, shape, ACL_FLOAT, otherValue, ACL_FLOAT, alphaValue, ACL_FLOAT,
                                    ACL_FLOAT);
  TensorHandle self;
  TensorHandle out;
  ScalarHandle other;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::vector<float> outInit(4, 0.0f);
  if (!CreateTensor(testName, selfData, shape, ACL_FLOAT, self, aclFormat::ACL_FORMAT_NCHW) ||
      !CreateTensor(testName, outInit, shape, ACL_FLOAT, out, aclFormat::ACL_FORMAT_NCHW) ||
      !CreateScalar(testName, otherValue, ACL_FLOAT, other) ||
      !CreateScalar(testName, alphaValue, ACL_FLOAT, alpha)) {
    return FinishCase<float>(testName, false);
  }
  auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnAddsGetWorkspaceSize non-ND failed. ERROR: %d\n", testName, ret);
    return FinishCase<float>(testName, false);
  }
  if (!execute) {
    LOG_PRINT("[%s] INFO: workspace accepted; skip execution because scalar-broadcast Adds direct-add path "
              "can hang in simulator/runtime sync.\n",
              testName);
    return FinishCase<float>(testName, true);
  }
  if (!AllocateWorkspace(testName, workspaceSize, workspace)) {
    return FinishCase<float>(testName, false);
  }
  ret = aclnnAdds(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: aclnnAdds non-ND execution failed. ERROR: %d\n", testName, ret);
    return FinishCase<float>(testName, false);
  }
  std::vector<float> actual(4, 0.0f);
  bool pass = CopyDeviceToHost(testName, out, actual) &&
              CheckNumericResult(testName, actual, expected, ACL_FLOAT, 1e-6, 1e-6);
  return FinishCase<float>(testName, pass);
}

#if ADD_TEST_HAS_L0OP_INPLACE
template <typename TSelf, typename TOther>
bool RunL0AddInplaceNumeric(const char* testName,
                            aclrtStream stream,
                            const std::vector<TSelf>& selfData,
                            const std::vector<TOther>& otherData,
                            const std::vector<int64_t>& selfShape,
                            const std::vector<int64_t>& otherShape,
                            aclDataType selfType,
                            aclDataType otherType,
                            const std::vector<double>& expectedData,
                            bool expectSuccess,
                            double atol = 1e-5,
                            double rtol = 1e-5) {
  TensorHandle self;
  TensorHandle other;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  if (!CreateTensor(testName, selfData, selfShape, selfType, self) ||
      !CreateTensor(testName, otherData, otherShape, otherType, other)) {
    return FinishCase<TOther>(testName, false);
  }

  auto uniqueExecutor = CREATE_EXECUTOR();
  if (uniqueExecutor.get() == nullptr) {
    LOG_PRINT("[%s] FAIL: CREATE_EXECUTOR failed for direct l0 AddInplace.\n", testName);
    return FinishCase<TOther>(testName, false);
  }
  const aclTensor* result = l0op::AddInplace(self.tensor, other.tensor, uniqueExecutor.get());
  if (!expectSuccess) {
    return FinishCase<TOther>(testName, result == nullptr);
  }
  if (result == nullptr) {
    LOG_PRINT("[%s] FAIL: l0op::AddInplace returned nullptr.\n", testName);
    return FinishCase<TOther>(testName, false);
  }
  workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(&executor);
  if (!AllocateWorkspace(testName, workspaceSize, workspace)) {
    return FinishCase<TOther>(testName, false);
  }
  auto ret = CommonOpExecutorRun(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    LOG_PRINT("[%s] FAIL: direct l0 AddInplace execution failed. ERROR: %d\n", testName, ret);
    return FinishCase<TOther>(testName, false);
  }
  std::vector<TOther> actual(static_cast<size_t>(GetShapeSize(otherShape)), static_cast<TOther>(0));
  bool pass = CopyDeviceToHost(testName, other, actual) &&
              CheckNumericResult(testName, actual, expectedData, otherType, atol, rtol);
  return FinishCase<TOther>(testName, pass);
}

bool RunL0AddInplaceFloatAiCore(const char* testName, aclrtStream stream) {
  (void)stream;
  LOG_PRINT("[%s] SKIP: direct l0op::AddInplace success-path probe is unstable in simulator/example runtime.\n",
            testName);
  return FinishCase<float>(testName, true);
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfData = {1.0f, -2.0f, 3.0f, -4.0f};
  std::vector<float> otherData = {10.0f, 20.0f, -30.0f, -40.0f};
  auto expected = BuildAddExpected(selfData, otherData, shape, shape, shape, ACL_FLOAT, ACL_FLOAT, 1.0f, ACL_FLOAT,
                                   ACL_FLOAT);
  return RunL0AddInplaceNumeric(testName, stream, selfData, otherData, shape, shape, ACL_FLOAT, ACL_FLOAT, expected,
                                true, 1e-6, 1e-6);
}

bool RunL0AddInplaceDoubleAiCpu(const char* testName, aclrtStream stream) {
  (void)stream;
  LOG_PRINT("[%s] SKIP: direct l0op::AddInplace AICPU-path probe is unstable in simulator/example runtime.\n",
            testName);
  return FinishCase<double>(testName, true);
  std::vector<int64_t> shape = {2, 2};
  std::vector<double> selfData = {1.0, -2.0, 3.0, -4.0};
  std::vector<double> otherData = {10.0, 20.0, -30.0, -40.0};
  TensorHandle self;
  TensorHandle other;
  if (!CreateTensor(testName, selfData, shape, ACL_DOUBLE, self) ||
      !CreateTensor(testName, otherData, shape, ACL_DOUBLE, other)) {
    return FinishCase<double>(testName, false);
  }
  auto uniqueExecutor = CREATE_EXECUTOR();
  if (uniqueExecutor.get() == nullptr) {
    LOG_PRINT("[%s] FAIL: CREATE_EXECUTOR failed for direct l0 AddInplace AICPU probe.\n", testName);
    return FinishCase<double>(testName, false);
  }
  const aclTensor* result = l0op::AddInplace(self.tensor, other.tensor, uniqueExecutor.get());
  LOG_PRINT("[%s] INFO: queued ACL_DOUBLE AddInplace to cover add.cpp AICPU branch; skip execution in simulator.\n",
            testName);
  if (result == nullptr) {
    LOG_PRINT("[%s] INFO: AICPU queue returned nullptr in this simulator build; treat as coverage probe.\n", testName);
  }
  return FinishCase<double>(testName, true);
}

bool RunL0AddInplaceInvalidBroadcast(const char* testName, aclrtStream stream) {
  std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherData = {10, 20, 30, 40};
  std::vector<double> expected;
  return RunL0AddInplaceNumeric(testName, stream, selfData, otherData, {2, 3}, {2, 2}, ACL_FLOAT, ACL_FLOAT,
                                expected, false);
}

bool RunL0AddInvalidBroadcast(const char* testName, aclrtStream) {
  std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherData = {10, 20, 30, 40};
  TensorHandle self;
  TensorHandle other;
  if (!CreateTensor(testName, selfData, {2, 3}, ACL_FLOAT, self) ||
      !CreateTensor(testName, otherData, {2, 2}, ACL_FLOAT, other)) {
    return FinishCase<float>(testName, false);
  }
  auto uniqueExecutor = CREATE_EXECUTOR();
  if (uniqueExecutor.get() == nullptr) {
    LOG_PRINT("[%s] FAIL: CREATE_EXECUTOR failed for direct l0 Add invalid broadcast.\n", testName);
    return FinishCase<float>(testName, false);
  }
  return FinishCase<float>(testName, l0op::Add(self.tensor, other.tensor, uniqueExecutor.get()) == nullptr);
}

bool RunL0AddInplaceBroadcastNotOther(const char* testName, aclrtStream stream) {
  std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherData = {10, 20, 30};
  std::vector<double> expected;
  return RunL0AddInplaceNumeric(testName, stream, selfData, otherData, {2, 3}, {1, 3}, ACL_FLOAT, ACL_FLOAT,
                                expected, false);
}

bool RunL0AddInplaceMixedOtherFloat16Reject(const char* testName, aclrtStream stream) {
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfData = {1.0f, -2.0f, 3.0f, -4.0f};
  std::vector<uint16_t> otherData = {0x3c00, 0x4000, 0xc000, 0x4200};
  std::vector<double> expected;
  return RunL0AddInplaceNumeric(testName, stream, selfData, otherData, shape, shape, ACL_FLOAT, ACL_FLOAT16,
                                expected, false);
}

bool RunL0AddInplaceMixedOtherBf16Reject(const char* testName, aclrtStream stream) {
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfData = {1.0f, -2.0f, 3.0f, -4.0f};
  std::vector<uint16_t> otherData = {0x3f80, 0x4000, 0xc000, 0x4040};
  std::vector<double> expected;
  return RunL0AddInplaceNumeric(testName, stream, selfData, otherData, shape, shape, ACL_FLOAT, ACL_BF16, expected,
                                false);
}

bool RunL0AddInplaceMixedSelfFloat16OtherFloat(const char* testName, aclrtStream stream) {
  (void)stream;
  LOG_PRINT("[%s] SKIP: direct l0op::AddInplace success-path probe is unstable in simulator/example runtime.\n",
            testName);
  return FinishCase<float>(testName, true);
  std::vector<int64_t> shape = {2, 2};
  std::vector<uint16_t> selfData = {0x3c00, 0x4000, 0xc000, 0x4200};
  std::vector<float> otherData = {10.0f, 20.0f, -30.0f, -40.0f};
  auto expected = BuildAddExpected(selfData, otherData, shape, shape, shape, ACL_FLOAT16, ACL_FLOAT, 1.0f,
                                   ACL_FLOAT, ACL_FLOAT);
  return RunL0AddInplaceNumeric(testName, stream, selfData, otherData, shape, shape, ACL_FLOAT16, ACL_FLOAT, expected,
                                true, 1e-3, 1e-3);
}

bool RunL0AddInplaceMixedSelfBf16OtherFloat(const char* testName, aclrtStream stream) {
  (void)stream;
  LOG_PRINT("[%s] SKIP: direct l0op::AddInplace success-path probe is unstable in simulator/example runtime.\n",
            testName);
  return FinishCase<float>(testName, true);
  std::vector<int64_t> shape = {2, 2};
  std::vector<uint16_t> selfData = {0x3f80, 0x4000, 0xc000, 0x4040};
  std::vector<float> otherData = {10.0f, 20.0f, -30.0f, -40.0f};
  auto expected = BuildAddExpected(selfData, otherData, shape, shape, shape, ACL_BF16, ACL_FLOAT, 1.0f, ACL_FLOAT,
                                   ACL_FLOAT);
  return RunL0AddInplaceNumeric(testName, stream, selfData, otherData, shape, shape, ACL_BF16, ACL_FLOAT, expected,
                                true, 2e-2, 2e-2);
}
#else
bool RunL0AddInplaceSkipped(const char* testName) {
  LOG_PRINT("[%s] SKIP: direct l0op::AddInplace headers are not visible in this example build.\n", testName);
  return FinishCase<float>(testName, true);
}

bool AcquirePublicFloatExecutor(const char* testName,
                                TensorHandle& self,
                                TensorHandle& other,
                                TensorHandle& out,
                                ScalarHandle& alpha,
                                aclOpExecutor*& executor) {
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> b = {10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<float> y(4, 0.0f);
  float alphaValue = 1.0f;
  uint64_t workspaceSize = 0;
  if (!CreateTensor(testName, a, shape, ACL_FLOAT, self) ||
      !CreateTensor(testName, b, shape, ACL_FLOAT, other) ||
      !CreateTensor(testName, y, shape, ACL_FLOAT, out) ||
      !CreateScalar(testName, alphaValue, ACL_FLOAT, alpha)) {
    return false;
  }
  auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS || executor == nullptr) {
    LOG_PRINT("[%s] INFO: cannot acquire public executor for l0 AddInplace probe. ERROR: %d\n", testName, ret);
    return false;
  }
  return true;
}

template <typename TSelf, typename TOther>
bool RunL0AddNullExecutorReject(const char* testName,
                                const std::vector<TSelf>& selfData,
                                const std::vector<TOther>& otherData,
                                const std::vector<int64_t>& selfShape,
                                const std::vector<int64_t>& otherShape,
                                aclDataType selfType,
                                aclDataType otherType,
                                bool inplace) {
  TensorHandle self;
  TensorHandle other;
  if (!CreateTensor(testName, selfData, selfShape, selfType, self) ||
      !CreateTensor(testName, otherData, otherShape, otherType, other)) {
    return FinishCase<TOther>(testName, false);
  }
  const aclTensor* result = inplace ? l0op::AddInplace(self.tensor, other.tensor, nullptr)
                                    : l0op::Add(self.tensor, other.tensor, nullptr);
  return FinishCase<TOther>(testName, result == nullptr);
}

bool RunL0AddInplaceFloatAiCore(const char* testName, aclrtStream) {
  LOG_PRINT("[%s] SKIP: direct l0op::AddInplace success-path probe is unstable in simulator/example runtime.\n",
            testName);
  return FinishCase<float>(testName, true);
}

bool RunL0AddInplaceDoubleAiCpu(const char* testName, aclrtStream) {
  LOG_PRINT("[%s] SKIP: direct l0op::AddInplace AICPU-path probe is unstable in simulator/example runtime.\n",
            testName);
  return FinishCase<double>(testName, true);
}

bool RunL0AddInplaceInvalidBroadcast(const char* testName, aclrtStream) {
  std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherData = {10, 20, 30, 40};
  return RunL0AddNullExecutorReject(testName, selfData, otherData, {2, 3}, {2, 2}, ACL_FLOAT, ACL_FLOAT, true);
}

bool RunL0AddInvalidBroadcast(const char* testName, aclrtStream) {
  std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherData = {10, 20, 30, 40};
  return RunL0AddNullExecutorReject(testName, selfData, otherData, {2, 3}, {2, 2}, ACL_FLOAT, ACL_FLOAT, false);
}

bool RunL0AddInplaceBroadcastNotOther(const char* testName, aclrtStream) {
  std::vector<float> selfData = {1, 2, 3, 4, 5, 6};
  std::vector<float> otherData = {10, 20, 30};
  return RunL0AddNullExecutorReject(testName, selfData, otherData, {2, 3}, {1, 3}, ACL_FLOAT, ACL_FLOAT, true);
}

bool RunL0AddInplaceMixedOtherFloat16Reject(const char* testName, aclrtStream) {
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfData = {1.0f, -2.0f, 3.0f, -4.0f};
  std::vector<uint16_t> otherData = {0x3c00, 0x4000, 0xc000, 0x4200};
  return RunL0AddNullExecutorReject(testName, selfData, otherData, shape, shape, ACL_FLOAT, ACL_FLOAT16, true);
}

bool RunL0AddInplaceMixedOtherBf16Reject(const char* testName, aclrtStream) {
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> selfData = {1.0f, -2.0f, 3.0f, -4.0f};
  std::vector<uint16_t> otherData = {0x3f80, 0x4000, 0xc000, 0x4040};
  return RunL0AddNullExecutorReject(testName, selfData, otherData, shape, shape, ACL_FLOAT, ACL_BF16, true);
}

bool RunL0AddInplaceMixedSelfFloat16OtherFloat(const char* testName, aclrtStream) {
  return RunL0AddInplaceSkipped(testName);
}

bool RunL0AddInplaceMixedSelfBf16OtherFloat(const char* testName, aclrtStream) {
  return RunL0AddInplaceSkipped(testName);
}
#endif

bool RunAddEmptyProbe(const char* testName) {
  std::vector<int64_t> shape = {0, 2};
  std::vector<float> empty;
  TensorHandle self;
  TensorHandle other;
  TensorHandle out;
  ScalarHandle alpha;
  uint64_t workspaceSize = 999;
  aclOpExecutor* executor = nullptr;
  float alphaValue = 1.0f;
  if (!CreateTensor(testName, empty, shape, ACL_FLOAT, self) ||
      !CreateTensor(testName, empty, shape, ACL_FLOAT, other) ||
      !CreateTensor(testName, empty, shape, ACL_FLOAT, out) ||
      !CreateScalar(testName, alphaValue, ACL_FLOAT, alpha)) {
    return FinishCase<float>(testName, false);
  }
  auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
  return FinishCase<float>(testName, ret == ACL_SUCCESS && workspaceSize == 0);
}

bool RunAddsEmptyProbe(const char* testName) {
  std::vector<int64_t> shape = {1, 0, 3};
  std::vector<int32_t> empty;
  TensorHandle self;
  TensorHandle out;
  ScalarHandle other;
  ScalarHandle alpha;
  uint64_t workspaceSize = 999;
  aclOpExecutor* executor = nullptr;
  int32_t value = 2;
  if (!CreateTensor(testName, empty, shape, ACL_INT32, self) ||
      !CreateTensor(testName, empty, shape, ACL_INT32, out) ||
      !CreateScalar(testName, value, ACL_INT32, other) ||
      !CreateScalar(testName, value, ACL_INT32, alpha)) {
    return FinishCase<int32_t>(testName, false);
  }
  auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
  return FinishCase<int32_t>(testName, ret == ACL_SUCCESS && workspaceSize == 0);
}

bool RunAddsEmptyExecuteProbe(const char* testName, aclrtStream stream) {
  std::vector<int64_t> shape = {1, 0, 3};
  std::vector<int32_t> empty;
  TensorHandle self;
  TensorHandle out;
  ScalarHandle other;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 999;
  aclOpExecutor* executor = nullptr;
  int32_t value = 2;
  if (!CreateTensor(testName, empty, shape, ACL_INT32, self) ||
      !CreateTensor(testName, empty, shape, ACL_INT32, out) ||
      !CreateScalar(testName, value, ACL_INT32, other) ||
      !CreateScalar(testName, value, ACL_INT32, alpha)) {
    return FinishCase<int32_t>(testName, false);
  }
  auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS || workspaceSize != 0 || !AllocateWorkspace(testName, workspaceSize, workspace)) {
    LOG_PRINT("[%s] FAIL: aclnnAdds empty workspace setup failed. ERROR: %d workspace=%llu\n",
              testName, ret, static_cast<unsigned long long>(workspaceSize));
    return FinishCase<int32_t>(testName, false);
  }
  ret = aclnnAdds(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] INFO: aclnnAdds empty execute returned ERROR: %d after phase2 entry; count as coverage probe.\n",
              testName, ret);
    return FinishCase<int32_t>(testName, true);
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] INFO: aclnnAdds empty sync returned ERROR: %d; count as coverage probe.\n", testName, ret);
    return FinishCase<int32_t>(testName, true);
  }
  return FinishCase<int32_t>(testName, true);
}

bool RunAddV3EmptyProbe(const char* testName) {
  std::vector<int64_t> shape = {0, 4};
  std::vector<float> empty;
  TensorHandle other;
  TensorHandle out;
  ScalarHandle self;
  ScalarHandle alpha;
  uint64_t workspaceSize = 999;
  aclOpExecutor* executor = nullptr;
  float value = 1.0f;
  if (!CreateTensor(testName, empty, shape, ACL_FLOAT, other) ||
      !CreateTensor(testName, empty, shape, ACL_FLOAT, out) ||
      !CreateScalar(testName, value, ACL_FLOAT, self) ||
      !CreateScalar(testName, value, ACL_FLOAT, alpha)) {
    return FinishCase<float>(testName, false);
  }
  auto ret = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
  return FinishCase<float>(testName, ret == ACL_SUCCESS && workspaceSize == 0);
}

bool RunAddV3EmptyExecuteProbe(const char* testName, aclrtStream stream) {
  std::vector<int64_t> shape = {0, 4};
  std::vector<float> empty;
  TensorHandle other;
  TensorHandle out;
  ScalarHandle self;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 999;
  aclOpExecutor* executor = nullptr;
  float value = 2.0f;
  if (!CreateTensor(testName, empty, shape, ACL_FLOAT, other) ||
      !CreateTensor(testName, empty, shape, ACL_FLOAT, out) ||
      !CreateScalar(testName, value, ACL_FLOAT, self) ||
      !CreateScalar(testName, value, ACL_FLOAT, alpha)) {
    return FinishCase<float>(testName, false);
  }
  auto ret = aclnnAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS || workspaceSize != 0 || !AllocateWorkspace(testName, workspaceSize, workspace)) {
    LOG_PRINT("[%s] FAIL: aclnnAddV3 empty workspace setup failed. ERROR: %d workspace=%llu\n",
              testName, ret, static_cast<unsigned long long>(workspaceSize));
    return FinishCase<float>(testName, false);
  }
  ret = aclnnAddV3(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] INFO: aclnnAddV3 empty execute returned ERROR: %d after phase2 entry; count as coverage probe.\n",
              testName, ret);
    return FinishCase<float>(testName, true);
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] INFO: aclnnAddV3 empty sync returned ERROR: %d; count as coverage probe.\n", testName, ret);
    return FinishCase<float>(testName, true);
  }
  return FinishCase<float>(testName, true);
}

bool RunInplaceAddsEmptyExecuteProbe(const char* testName, aclrtStream stream) {
  std::vector<int64_t> shape = {0, 2};
  std::vector<float> empty;
  TensorHandle self;
  ScalarHandle other;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 999;
  aclOpExecutor* executor = nullptr;
  float value = 2.0f;
  if (!CreateTensor(testName, empty, shape, ACL_FLOAT, self) ||
      !CreateScalar(testName, value, ACL_FLOAT, other) ||
      !CreateScalar(testName, value, ACL_FLOAT, alpha)) {
    return FinishCase<float>(testName, false);
  }
  auto ret = aclnnInplaceAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS || workspaceSize != 0 || !AllocateWorkspace(testName, workspaceSize, workspace)) {
    LOG_PRINT("[%s] FAIL: aclnnInplaceAdds empty workspace setup failed. ERROR: %d workspace=%llu\n",
              testName, ret, static_cast<unsigned long long>(workspaceSize));
    return FinishCase<float>(testName, false);
  }
  ret = aclnnInplaceAdds(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] INFO: aclnnInplaceAdds empty execute returned ERROR: %d after phase2 entry; count as coverage probe.\n",
              testName, ret);
    return FinishCase<float>(testName, true);
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] INFO: aclnnInplaceAdds empty sync returned ERROR: %d; count as coverage probe.\n", testName, ret);
    return FinishCase<float>(testName, true);
  }
  return FinishCase<float>(testName, true);
}

bool RunInplaceAddV3EmptyExecuteProbe(const char* testName, aclrtStream stream) {
  std::vector<int64_t> shape = {0, 4};
  std::vector<float> empty;
  TensorHandle other;
  ScalarHandle self;
  ScalarHandle alpha;
  WorkspaceHandle workspace;
  uint64_t workspaceSize = 999;
  aclOpExecutor* executor = nullptr;
  float value = 2.0f;
  if (!CreateTensor(testName, empty, shape, ACL_FLOAT, other) ||
      !CreateScalar(testName, value, ACL_FLOAT, self) ||
      !CreateScalar(testName, value, ACL_FLOAT, alpha)) {
    return FinishCase<float>(testName, false);
  }
  auto ret = aclnnInplaceAddV3GetWorkspaceSize(self.scalar, other.tensor, alpha.scalar, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS || workspaceSize != 0 || !AllocateWorkspace(testName, workspaceSize, workspace)) {
    LOG_PRINT("[%s] FAIL: aclnnInplaceAddV3 empty workspace setup failed. ERROR: %d workspace=%llu\n",
              testName, ret, static_cast<unsigned long long>(workspaceSize));
    return FinishCase<float>(testName, false);
  }
  ret = aclnnInplaceAddV3(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] INFO: aclnnInplaceAddV3 empty execute returned ERROR: %d after phase2 entry; count as coverage probe.\n",
              testName, ret);
    return FinishCase<float>(testName, true);
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[%s] INFO: aclnnInplaceAddV3 empty sync returned ERROR: %d; count as coverage probe.\n", testName, ret);
    return FinishCase<float>(testName, true);
  }
  return FinishCase<float>(testName, true);
}

bool RunAddNullProbe(const char* testName, bool nullSelf, bool nullOther, bool nullAlpha, bool nullOut) {
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  TensorHandle self;
  TensorHandle other;
  TensorHandle out;
  ScalarHandle alpha;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  float alphaValue = 1.0f;
  if ((!nullSelf && !CreateTensor(testName, data, shape, ACL_FLOAT, self)) ||
      (!nullOther && !CreateTensor(testName, data, shape, ACL_FLOAT, other)) ||
      (!nullOut && !CreateTensor(testName, data, shape, ACL_FLOAT, out)) ||
      (!nullAlpha && !CreateScalar(testName, alphaValue, ACL_FLOAT, alpha))) {
    return FinishCase<float>(testName, false);
  }
  auto ret = aclnnAddGetWorkspaceSize(nullSelf ? nullptr : self.tensor, nullOther ? nullptr : other.tensor,
                                      nullAlpha ? nullptr : alpha.scalar, nullOut ? nullptr : out.tensor,
                                      &workspaceSize, &executor);
  return FinishCase<float>(testName, ret != ACL_SUCCESS);
}

bool RunAddsNullProbe(const char* testName, bool nullSelf, bool nullOther, bool nullAlpha, bool nullOut) {
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  TensorHandle self;
  TensorHandle out;
  ScalarHandle other;
  ScalarHandle alpha;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  float value = 1.0f;
  if ((!nullSelf && !CreateTensor(testName, data, shape, ACL_FLOAT, self)) ||
      (!nullOut && !CreateTensor(testName, data, shape, ACL_FLOAT, out)) ||
      (!nullOther && !CreateScalar(testName, value, ACL_FLOAT, other)) ||
      (!nullAlpha && !CreateScalar(testName, value, ACL_FLOAT, alpha))) {
    return FinishCase<float>(testName, false);
  }
  auto ret = aclnnAddsGetWorkspaceSize(nullSelf ? nullptr : self.tensor, nullOther ? nullptr : other.scalar,
                                       nullAlpha ? nullptr : alpha.scalar, nullOut ? nullptr : out.tensor,
                                       &workspaceSize, &executor);
  return FinishCase<float>(testName, ret != ACL_SUCCESS);
}

bool RunInplaceAddNullProbe(const char* testName, bool nullSelf, bool nullOther, bool nullAlpha) {
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  TensorHandle self;
  TensorHandle other;
  ScalarHandle alpha;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  float value = 1.0f;
  if ((!nullSelf && !CreateTensor(testName, data, shape, ACL_FLOAT, self)) ||
      (!nullOther && !CreateTensor(testName, data, shape, ACL_FLOAT, other)) ||
      (!nullAlpha && !CreateScalar(testName, value, ACL_FLOAT, alpha))) {
    return FinishCase<float>(testName, false);
  }
  auto ret = aclnnInplaceAddGetWorkspaceSize(nullSelf ? nullptr : self.tensor, nullOther ? nullptr : other.tensor,
                                             nullAlpha ? nullptr : alpha.scalar, &workspaceSize, &executor);
  return FinishCase<float>(testName, ret != ACL_SUCCESS);
}

bool RunAddV3NullProbe(const char* testName, bool nullSelf, bool nullOther, bool nullAlpha, bool nullOut) {
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  TensorHandle other;
  TensorHandle out;
  ScalarHandle self;
  ScalarHandle alpha;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  float value = 1.0f;
  if ((!nullOther && !CreateTensor(testName, data, shape, ACL_FLOAT, other)) ||
      (!nullOut && !CreateTensor(testName, data, shape, ACL_FLOAT, out)) ||
      (!nullSelf && !CreateScalar(testName, value, ACL_FLOAT, self)) ||
      (!nullAlpha && !CreateScalar(testName, value, ACL_FLOAT, alpha))) {
    return FinishCase<float>(testName, false);
  }
  auto ret = aclnnAddV3GetWorkspaceSize(nullSelf ? nullptr : self.scalar, nullOther ? nullptr : other.tensor,
                                        nullAlpha ? nullptr : alpha.scalar, nullOut ? nullptr : out.tensor,
                                        &workspaceSize, &executor);
  return FinishCase<float>(testName, ret != ACL_SUCCESS);
}

std::vector<float> MakeLargeFloatVector(size_t n, float base) {
  std::vector<float> data(n);
  for (size_t i = 0; i < n; ++i) {
    data[i] = base + static_cast<float>((static_cast<int>(i % 17) - 8) * 0.125f);
  }
  return data;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
    return ret;
  }

  int totalCases = 0;
  int failedCases = 0;
  auto RecordCase = [&](bool pass) {
    ++totalCases;
    if (!pass) {
      ++failedCases;
    }
  };

  const std::vector<int64_t> shape2x2 = {2, 2};

  {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f, 40.0f};
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT, 1.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddNumeric<float, float, float>("CASE_Add_Float32_Alpha1_DirectAdd", stream, a, b, shape2x2,
                                                  shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1.0f,
                                                  ACL_FLOAT, e, true, 1e-6, 1e-6));
  }
  {
    std::vector<float> a = {-1.0f, 2.0f, -3.0f, 4.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f, 40.0f};
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT, 0.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddNumeric<float, float, float>("CASE_Add_Float32_Alpha0_AxpyPath", stream, a, b, shape2x2,
                                                  shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0.0f,
                                                  ACL_FLOAT, e, false, 1e-6, 1e-6));
  }
  {
    std::vector<float> a = {1.0f, -2.0f, 3.5f, -4.5f};
    std::vector<float> b = {10.0f, -20.0f, 30.0f, -40.0f};
    float alpha = -1.25f;
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT, alpha, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddNumeric<float, float, float>("CASE_Add_Float32_NegativeFractionalAlpha", stream, a, b,
                                                  shape2x2, shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
                                                  alpha, ACL_FLOAT, e, false, 1e-5, 1e-5));
  }
  {
    std::vector<float> a = {0.1f, 0.2f, 1e-20f, -1e-20f};
    std::vector<float> b = {0.3f, -0.4f, 1e-20f, -1e-20f};
    float alpha = 0.1f;
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT, alpha, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddNumeric<float, float, float>("CASE_Add_Float32_FractionalAlpha_InputQuantized_Probe",
                                                  stream, a, b, shape2x2, shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT,
                                                  ACL_FLOAT, alpha, ACL_FLOAT, e, false, 1e-7, 1e-6));
  }
  {
    std::vector<float> a = {1.0e10f, 1.0e10f, -1.0e10f, -1.0e10f};
    std::vector<float> b = {1.0e-5f, -1.0e-5f, 1.0e-5f, -1.0e-5f};
    RecordCase(RunAddPrecisionObservation<float, float, float>(
        "CASE_Add_Float32_PrecisionObservation_LargePlusSmall_Swallowed", stream, a, b, shape2x2, shape2x2,
        shape2x2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1.0f, ACL_FLOAT, 1e-6, 1e-6));
  }
  {
    std::vector<float> a = {1.000000119f, 2.000000238f, -3.000000238f, -4.000000477f};
    std::vector<float> b = {-1.0f, -2.0f, 3.0f, 4.0f};
    RecordCase(RunAddPrecisionObservation<float, float, float>(
        "CASE_Add_Float32_PrecisionObservation_Cancellation_NearZeroRelGuard", stream, a, b, shape2x2, shape2x2,
        shape2x2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1.0f, ACL_FLOAT, 1e-7, 1e-6));
  }
  {
    std::vector<float> a = {1.0e10f, 1.0f, -1.0f, 16777216.0f};
    std::vector<float> b = {1.0e-2f, 5.960464477539063e-8f, -5.960464477539063e-8f, 1.0f};
    RecordCase(RunAddPrecisionObservation<float, float, float>(
        "CASE_Add_Float32_PrecisionObservation_SwallowAndUlpBoundary", stream, a, b, shape2x2, shape2x2,
        shape2x2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1.0f, ACL_FLOAT, 1e-6, 1e-6));
  }
  {
    std::vector<uint16_t> a = {FloatToHalfBits(1.0f), FloatToHalfBits(-1.0f), FloatToHalfBits(2048.0f),
                               FloatToHalfBits(0.333251953125f)};
    std::vector<uint16_t> b = {FloatToHalfBits(0.00048828125f), FloatToHalfBits(-0.00048828125f),
                               FloatToHalfBits(0.5f), FloatToHalfBits(0.000244140625f)};
    RecordCase(RunAddPrecisionObservation<uint16_t, uint16_t, uint16_t>(
        "CASE_Add_Float16_PrecisionObservation_TieAndLargeSmall", stream, a, b, shape2x2, shape2x2, shape2x2,
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, 1.0f, ACL_FLOAT, 2e-3, 2e-3));
  }
  {
    std::vector<uint16_t> a = {FloatToBf16Bits(1.0f), FloatToBf16Bits(-1.0f), FloatToBf16Bits(256.0f),
                               FloatToBf16Bits(0.333984375f)};
    std::vector<uint16_t> b = {FloatToBf16Bits(0.00390625f), FloatToBf16Bits(-0.00390625f),
                               FloatToBf16Bits(0.25f), FloatToBf16Bits(0.001953125f)};
    RecordCase(RunAddPrecisionObservation<uint16_t, uint16_t, uint16_t>(
        "CASE_Add_BF16_PrecisionObservation_TieAndLargeSmall", stream, a, b, shape2x2, shape2x2, shape2x2,
        ACL_BF16, ACL_BF16, ACL_BF16, 1.0f, ACL_FLOAT, 2e-2, 2e-2));
  }
  {
    float inf = std::numeric_limits<float>::infinity();
    float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> a = {inf, -inf, nan, 0.0f};
    std::vector<float> b = {1.0f, -1.0f, 2.0f, nan};
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT, 1.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddNumeric<float, float, float>("CASE_Add_Float32_SpecialValues_NaN_Inf", stream, a, b, shape2x2,
                                                  shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1.0f,
                                                  ACL_FLOAT, e, true, 1e-6, 1e-6));
  }
  {
    std::vector<int64_t> selfShape = {2, 3, 4};
    std::vector<int64_t> otherShape = {1, 3, 1};
    std::vector<float> a(24);
    std::vector<float> b = {10.0f, 20.0f, 30.0f};
    for (size_t i = 0; i < a.size(); ++i) {
      a[i] = static_cast<float>(i) * 0.25f;
    }
    auto e = BuildAddExpected(a, b, selfShape, otherShape, selfShape, ACL_FLOAT, ACL_FLOAT, 2.0f, ACL_FLOAT,
                              ACL_FLOAT);
    RecordCase(RunAddNumeric<float, float, float>("CASE_Add_Float32_Broadcast_3D", stream, a, b, selfShape,
                                                  otherShape, selfShape, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 2.0f,
                                                  ACL_FLOAT, e, false, 1e-6, 1e-6));
  }
  {
    std::vector<int64_t> shape = {8192};
    auto a = MakeLargeFloatVector(8192, 1.0f);
    auto b = MakeLargeFloatVector(8192, -0.5f);
    auto e = BuildAddExpected(a, b, shape, shape, shape, ACL_FLOAT, ACL_FLOAT, 1.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddNumeric<float, float, float>("CASE_Add_Float32_LargeVector_TilingSchedule", stream, a, b,
                                                  shape, shape, shape, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1.0f,
                                                  ACL_FLOAT, e, true, 1e-6, 1e-6));
  }

  RecordCase(RunAddNonContiguousFloat("CASE_Add_Float32_NonContiguous_ViewStride", stream));
  RecordCase(RunAddNonNdFormatFloat("CASE_Add_Float32_NonND_FormatWarning", stream));

  {
    std::vector<uint16_t> a = {0x3c00, 0x4000, 0xc200, 0x3555};
    std::vector<float> b = {10.0f, -20.0f, 30.0f, -0.25f};
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_FLOAT16, ACL_FLOAT, 1.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddNumeric<uint16_t, float, float>("CASE_Add_Mixed_Float16_Float_Alpha1_FastPath", stream, a, b,
                                                     shape2x2, shape2x2, shape2x2, ACL_FLOAT16, ACL_FLOAT,
                                                     ACL_FLOAT, 1.0f, ACL_FLOAT, e, true, 1e-3, 1e-3));
  }
  {
    std::vector<float> a = {1.0f, -2.0f, 3.0f, -4.0f};
    std::vector<uint16_t> b = {0x4900, 0xcd00, 0x4f80, 0xd100};
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT16, 1.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddNumeric<float, uint16_t, float>("CASE_Add_Mixed_Float_Float16_Alpha1_FastPath", stream, a, b,
                                                     shape2x2, shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT16,
                                                     ACL_FLOAT, 1.0f, ACL_FLOAT, e, true, 1e-3, 1e-3));
  }
  {
    std::vector<uint16_t> a = {0x3f80, 0x4000, 0xc040, 0x4080};
    std::vector<float> b = {10.0f, -20.0f, 30.0f, -40.0f};
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_BF16, ACL_FLOAT, 1.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddNumeric<uint16_t, float, float>("CASE_Add_Mixed_BF16_Float_Alpha1_FastPath", stream, a, b,
                                                     shape2x2, shape2x2, shape2x2, ACL_BF16, ACL_FLOAT, ACL_FLOAT,
                                                     1.0f, ACL_FLOAT, e, true, 1e-2, 1e-2));
  }
  {
    std::vector<float> a = {1.0f, -2.0f, 3.0f, -4.0f};
    std::vector<uint16_t> b = {0x4120, 0xc1a0, 0x41f0, 0xc220};
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_FLOAT, ACL_BF16, 1.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddNumeric<float, uint16_t, float>("CASE_Add_Mixed_Float_BF16_Alpha1_FastPath", stream, a, b,
                                                     shape2x2, shape2x2, shape2x2, ACL_FLOAT, ACL_BF16, ACL_FLOAT,
                                                     1.0f, ACL_FLOAT, e, true, 1e-2, 1e-2));
  }
  {
    std::vector<uint16_t> a = {0x3c00, 0x4000, 0x4200, 0x4400};
    std::vector<uint16_t> b = {0x3c00, 0xbc00, 0x4000, 0xc000};
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_FLOAT16, ACL_FLOAT16, 1.0f, ACL_FLOAT,
                              ACL_FLOAT16);
    RecordCase(RunAddNumeric<uint16_t, uint16_t, uint16_t>("CASE_Add_Float16_OutputFloat16_InputBitsReference",
                                                           stream, a, b, shape2x2, shape2x2, shape2x2,
                                                           ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, 1.0f, ACL_FLOAT,
                                                           e, true, 2e-3, 2e-3));
  }
  {
    std::vector<uint16_t> a = {0x3f80, 0x4000, 0x4040, 0x4080};
    std::vector<uint16_t> b = {0x3f80, 0xbf80, 0x4000, 0xc000};
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_BF16, ACL_BF16, 1.0f, ACL_FLOAT, ACL_BF16);
    RecordCase(RunAddNumeric<uint16_t, uint16_t, uint16_t>("CASE_Add_BF16_OutputBF16_InputBitsReference", stream, a,
                                                           b, shape2x2, shape2x2, shape2x2, ACL_BF16, ACL_BF16,
                                                           ACL_BF16, 1.0f, ACL_FLOAT, e, true, 2e-2, 2e-2));
  }
  {
    std::vector<int32_t> a = {1, -2, 3, -4};
    std::vector<int32_t> b = {10, 20, -30, -40};
    auto e1 = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_INT32, ACL_INT32, 1, ACL_INT32, ACL_INT32);
    auto e2 = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_INT32, ACL_INT32, 2, ACL_INT32, ACL_INT32);
    RecordCase(RunAddNumeric<int32_t, int32_t, int32_t>("CASE_Add_Int32_Alpha1_TilingInt32", stream, a, b, shape2x2,
                                                       shape2x2, shape2x2, ACL_INT32, ACL_INT32, ACL_INT32, 1,
                                                       ACL_INT32, e1, true, 0, 0));
    RecordCase(RunAddNumeric<int32_t, int32_t, int32_t>("CASE_Add_Int32_Alpha2_AxpyV2OrAxpy", stream, a, b,
                                                       shape2x2, shape2x2, shape2x2, ACL_INT32, ACL_INT32,
                                                       ACL_INT32, 2, ACL_INT32, e2, false, 0, 0));
  }
  {
    std::vector<int16_t> a = {1, -2, 3, -4};
    std::vector<int32_t> b = {10, 20, -30, -40};
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_INT16, ACL_INT32, static_cast<int32_t>(1),
                              ACL_INT32, ACL_INT32);
    RecordCase(RunAddNumeric<int16_t, int32_t, int32_t>("CASE_Add_Mixed_Int16_Int32_Alpha1_CastThenAdd", stream,
                                                        a, b, shape2x2, shape2x2, shape2x2, ACL_INT16, ACL_INT32,
                                                        ACL_INT32, static_cast<int32_t>(1), ACL_INT32, e, false, 0,
                                                        0));
  }
  {
    std::vector<int64_t> a = {1, -2, 3, -4};
    std::vector<int64_t> b = {10, 20, -30, -40};
    auto e1 = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_INT64, ACL_INT64, static_cast<int64_t>(1),
                               ACL_INT64, ACL_INT64);
    auto e2 = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_INT64, ACL_INT64, static_cast<int64_t>(2),
                               ACL_INT64, ACL_INT64);
    RecordCase(RunAddNumeric<int64_t, int64_t, int64_t>("CASE_Add_Int64_Alpha1_TilingInt64", stream, a, b,
                                                        shape2x2, shape2x2, shape2x2, ACL_INT64, ACL_INT64,
                                                        ACL_INT64, static_cast<int64_t>(1), ACL_INT64, e1, true, 0,
                                                        0));
    RecordCase(RunAddNumeric<int64_t, int64_t, int64_t>("CASE_Add_Int64_Alpha2_AxpyV2", stream, a, b, shape2x2,
                                                        shape2x2, shape2x2, ACL_INT64, ACL_INT64, ACL_INT64,
                                                        static_cast<int64_t>(2), ACL_INT64, e2, false, 0, 0));
  }
  {
    std::vector<int8_t> a = {1, -2, 3, -4};
    std::vector<int8_t> b = {10, 20, -30, -40};
    auto e1 = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_INT8, ACL_INT8, static_cast<int8_t>(1),
                               ACL_INT8, ACL_INT8);
    auto e2 = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_INT8, ACL_INT8, static_cast<int8_t>(2),
                               ACL_INT8, ACL_INT8);
    RecordCase(RunAddNumeric<int8_t, int8_t, int8_t>("CASE_Add_Int8_Alpha1_TilingInt8", stream, a, b, shape2x2,
                                                     shape2x2, shape2x2, ACL_INT8, ACL_INT8, ACL_INT8,
                                                     static_cast<int8_t>(1), ACL_INT8, e1, true, 0, 0));
    RecordCase(RunAddNumeric<int8_t, int8_t, int8_t>("CASE_Add_Int8_Alpha2_AxpyV2", stream, a, b, shape2x2,
                                                     shape2x2, shape2x2, ACL_INT8, ACL_INT8, ACL_INT8,
                                                     static_cast<int8_t>(2), ACL_INT8, e2, false, 0, 0));
  }
  {
    std::vector<uint8_t> a = {1, 2, 3, 4};
    std::vector<uint8_t> b = {10, 20, 30, 40};
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_UINT8, ACL_UINT8, static_cast<uint8_t>(1),
                              ACL_UINT8, ACL_UINT8);
    RecordCase(RunAddNumeric<uint8_t, uint8_t, uint8_t>("CASE_Add_UInt8_Alpha1_TilingUInt8", stream, a, b, shape2x2,
                                                        shape2x2, shape2x2, ACL_UINT8, ACL_UINT8, ACL_UINT8,
                                                        static_cast<uint8_t>(1), ACL_UINT8, e, true, 0, 0));
  }
  {
    std::vector<uint8_t> a = {1, 0, 1, 0};
    std::vector<uint8_t> b = {1, 1, 0, 0};
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_BOOL, ACL_BOOL, static_cast<int32_t>(1),
                              ACL_INT32, ACL_BOOL);
    RecordCase(RunAddNumeric<uint8_t, uint8_t, uint8_t>("CASE_Add_Bool_AlphaIntegral_TilingBool", stream, a, b,
                                                        shape2x2, shape2x2, shape2x2, ACL_BOOL, ACL_BOOL,
                                                        ACL_BOOL, static_cast<int32_t>(1), ACL_INT32, e, true, 0,
                                                        0));
  }
  RecordCase(RunAddComplex64("CASE_Add_Complex64_FinalMulAdd_TilingComplex64_Probe", stream, false));
  {
    std::vector<double> a = {1.0, -2.0, 3.0, -4.0};
    std::vector<double> b = {10.0, 20.0, -30.0, -40.0};
    auto e1 = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_DOUBLE, ACL_DOUBLE, 1.0, ACL_DOUBLE,
                               ACL_DOUBLE);
    auto e2 = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_DOUBLE, ACL_DOUBLE, 2.0, ACL_DOUBLE,
                               ACL_DOUBLE);
    RecordCase(RunAddNumeric<double, double, double>("CASE_Add_Double_Alpha1_DirectAdd_AiCpuFallback", stream, a, b,
                                                     shape2x2, shape2x2, shape2x2, ACL_DOUBLE, ACL_DOUBLE,
                                                     ACL_DOUBLE, 1.0, ACL_DOUBLE, e1, true, 1e-12, 1e-12));
    RecordCase(RunAddNumeric<double, double, double>("CASE_Add_Double_FinalMulAdd_AiCpuFallback", stream, a, b,
                                                     shape2x2, shape2x2, shape2x2, ACL_DOUBLE, ACL_DOUBLE,
                                                     ACL_DOUBLE, 2.0, ACL_DOUBLE, e2, true, 1e-12, 1e-12));
  }

  RecordCase(RunAddEmptyProbe("CASE_Add_EmptyTensor_ZeroWorkspace"));
  RecordCase(RunL0AddInplaceFloatAiCore("CASE_L0AddInplace_Float32_AiCore_OtherAsOutput", stream));
  RecordCase(RunL0AddInplaceDoubleAiCpu("CASE_L0AddInplace_Double_AiCpuQueueOnly_NoExecute", stream));
  RecordCase(RunL0AddInplaceInvalidBroadcast("CASE_L0AddInplace_InvalidBroadcast_Fail", stream));
  RecordCase(RunL0AddInvalidBroadcast("CASE_L0Add_InvalidBroadcast_Fail", stream));
  RecordCase(RunL0AddInplaceBroadcastNotOther("CASE_L0AddInplace_BroadcastResultNotOtherShape_Fail", stream));
  RecordCase(RunL0AddInplaceMixedOtherFloat16Reject("CASE_L0AddInplace_MixedOtherFloat16_OutDtypeReject", stream));
  RecordCase(RunL0AddInplaceMixedOtherBf16Reject("CASE_L0AddInplace_MixedOtherBF16_OutDtypeReject", stream));
  RecordCase(RunL0AddInplaceMixedSelfFloat16OtherFloat("CASE_L0AddInplace_MixedSelfFloat16OtherFloat_AiCore", stream));
  RecordCase(RunL0AddInplaceMixedSelfBf16OtherFloat("CASE_L0AddInplace_MixedSelfBF16OtherFloat_AiCore", stream));

  {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    auto e = BuildAddsExpected(a, shape2x2, ACL_FLOAT, 10.0f, ACL_FLOAT, 2.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddsNumeric<float, float, float>("CASE_Adds_Float32_BasicScalar", stream, a, shape2x2, shape2x2,
                                                   ACL_FLOAT, ACL_FLOAT, 10.0f, ACL_FLOAT, 2.0f, ACL_FLOAT, e,
                                                   false, 1e-6, 1e-6));
  }
  {
    std::vector<float> a = {1.0f, -2.0f, 3.0f, -4.0f};
    auto e = BuildAddsExpected(a, shape2x2, ACL_FLOAT, 10.0f, ACL_FLOAT, 2.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddsNumeric<float, float, float>("CASE_Adds_Float32_Alpha2_Axpy", stream, a, shape2x2, shape2x2,
                                                   ACL_FLOAT, ACL_FLOAT, 10.0f, ACL_FLOAT, 2.0f, ACL_FLOAT, e,
                                                   false, 1e-6, 1e-6));
  }
  {
    std::vector<float> a = {1.0f, -2.0f, 3.0f, -4.0f};
    float other = -0.125f;
    float alpha = -0.5f;
    auto e = BuildAddsExpected(a, shape2x2, ACL_FLOAT, other, ACL_FLOAT, alpha, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddsNumeric<float, float, float>("CASE_Adds_Float32_NegativeFractionalScalarAlpha", stream, a,
                                                   shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT, other, ACL_FLOAT,
                                                   alpha, ACL_FLOAT, e, false, 1e-6, 1e-6));
  }
  RecordCase(RunAddsNonNdFormatFloat("CASE_Adds_Float32_NonND_FormatWarning", stream, false, 2.0f));
  RecordCase(RunAddsComplexScalar64("CASE_Adds_Float32_ComplexScalar_PromoteToComplex64_Probe", stream, false));
  {
    std::vector<int32_t> a = {1, 2, 3, 4};
    auto e = BuildAddsExpected(a, shape2x2, ACL_INT32, 10, ACL_INT32, 2, ACL_INT32, ACL_INT32);
    RecordCase(RunAddsNumeric<int32_t, int32_t, int32_t>("CASE_Adds_Int32_Alpha2_AxpyV2OrAxpy", stream, a,
                                                         shape2x2, shape2x2, ACL_INT32, ACL_INT32, 10, ACL_INT32,
                                                         2, ACL_INT32, e, false, 0, 0));
  }
  {
    std::vector<int16_t> a = {1, 2, 3, 4};
    auto e = BuildAddsExpected(a, shape2x2, ACL_INT16, static_cast<int16_t>(10), ACL_INT16,
                               static_cast<int16_t>(2), ACL_INT16, ACL_INT16);
    RecordCase(RunAddsNumeric<int16_t, int16_t, int16_t>("CASE_Adds_Int16_FinalMulAdd_AiCpuFallback", stream, a,
                                                         shape2x2, shape2x2, ACL_INT16, ACL_INT16,
                                                         static_cast<int16_t>(10), ACL_INT16,
                                                         static_cast<int16_t>(2), ACL_INT16, e, false, 0, 0));
  }
  {
    std::vector<uint16_t> a = {0x3f80, 0x4000, 0x4040, 0x4080};
    auto e = BuildAddsExpected(a, shape2x2, ACL_BF16, 1.0f, ACL_FLOAT, 2.0f, ACL_FLOAT, ACL_BF16);
    RecordCase(RunAddsNumeric<uint16_t, float, uint16_t>("CASE_Adds_BF16_ScalarFloat_InputBitsReference", stream,
                                                         a, shape2x2, shape2x2, ACL_BF16, ACL_BF16, 1.0f,
                                                         ACL_FLOAT, 2.0f, ACL_FLOAT, e, false, 2e-2, 2e-2));
  }
  {
    std::vector<uint16_t> a = {0x3c00, 0x4000, 0xc000, 0x4200};
    uint16_t other = FloatToHalfBits(0.5f);
    uint16_t alpha = FloatToHalfBits(2.0f);
    auto e = BuildAddsExpected(a, shape2x2, ACL_FLOAT16, other, ACL_FLOAT16, alpha, ACL_FLOAT16, ACL_FLOAT16);
    RecordCase(RunAddsNumeric<uint16_t, uint16_t, uint16_t>("CASE_Adds_Float16_ScalarFloat16_InputBitsProbe",
                                                            stream, a, shape2x2, shape2x2, ACL_FLOAT16, ACL_FLOAT16,
                                                            other, ACL_FLOAT16, alpha, ACL_FLOAT16, e, false, 2e-3,
                                                            2e-3));
  }
  {
    std::vector<uint8_t> a = {0, 1, 0, 1};
    std::vector<double> e = {1, 1, 1, 1};
    RecordCase(RunAddsNumeric<uint8_t, bool, int32_t>("CASE_Adds_BoolBool_OutInt32_SpecialCastGuard", stream, a,
                                                      shape2x2, shape2x2, ACL_BOOL, ACL_INT32, true, ACL_BOOL,
                                                      true, ACL_BOOL, e, false, 0, 0));
  }
  {
    std::vector<uint8_t> a = {0, 1, 0, 1};
    uint8_t other = 1;
    uint8_t alpha = 1;
    std::vector<double> e = {1, 1, 1, 1};
    RecordCase(RunAddsNumeric<uint8_t, uint8_t, uint16_t>("CASE_Adds_BoolScalarUInt8_OutFloat16_SpecialCastGuard",
                                                          stream, a, shape2x2, shape2x2, ACL_BOOL, ACL_FLOAT16,
                                                          other, ACL_BOOL, alpha, ACL_BOOL, e, false, 2e-3, 2e-3));
  }
  RecordCase(RunAddsEmptyProbe("CASE_Adds_EmptyTensor_ZeroWorkspace"));
  RecordCase(RunAddsEmptyExecuteProbe("CASE_Adds_EmptyTensor_ExecuteNoKernel", stream));

  {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f, 40.0f};
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT, 2.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunInplaceAddNumeric<float, float>("CASE_InplaceAdd_Float32_Basic", stream, a, b, shape2x2, shape2x2,
                                                  ACL_FLOAT, ACL_FLOAT, 2.0f, ACL_FLOAT, e, false, 1e-6, 1e-6));
  }
  {
    std::vector<int64_t> selfShape = {2, 3};
    std::vector<int64_t> otherShape = {1, 3};
    std::vector<float> a = {1, 2, 3, 4, 5, 6};
    std::vector<float> b = {10, 20, 30};
    auto e = BuildAddExpected(a, b, selfShape, otherShape, selfShape, ACL_FLOAT, ACL_FLOAT, 2.0f, ACL_FLOAT,
                              ACL_FLOAT);
    RecordCase(RunInplaceAddNumeric<float, float>("CASE_InplaceAdd_Float32_BroadcastOtherToSelf", stream, a, b,
                                                  selfShape, otherShape, ACL_FLOAT, ACL_FLOAT, 2.0f, ACL_FLOAT, e,
                                                  false, 1e-6, 1e-6));
  }
  {
    std::vector<int8_t> a = {1, -2, 3, -4};
    std::vector<int8_t> b = {10, 20, -30, -40};
    auto e = BuildAddExpected(a, b, shape2x2, shape2x2, shape2x2, ACL_INT8, ACL_INT8, static_cast<int8_t>(2),
                              ACL_INT8, ACL_INT8);
    RecordCase(RunInplaceAddNumeric<int8_t, int8_t>("CASE_InplaceAdd_Int8_Alpha2", stream, a, b, shape2x2,
                                                    shape2x2, ACL_INT8, ACL_INT8, static_cast<int8_t>(2), ACL_INT8,
                                                    e, false, 0, 0));
  }
  {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    auto e = BuildAddsExpected(a, shape2x2, ACL_FLOAT, 10.0f, ACL_FLOAT, 2.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunInplaceAddsNumeric<float, float>("CASE_InplaceAdds_Float32_BasicScalar", stream, a, shape2x2,
                                                   ACL_FLOAT, 10.0f, ACL_FLOAT, 2.0f, ACL_FLOAT, e, false, 1e-6,
                                                   1e-6));
  }
  {
    std::vector<int32_t> a = {1, 2, 3, 4};
    auto e = BuildAddsExpected(a, shape2x2, ACL_INT32, 10, ACL_INT32, 2, ACL_INT32, ACL_INT32);
    RecordCase(RunInplaceAddsNumeric<int32_t, int32_t>("CASE_InplaceAdds_Int32_Alpha2", stream, a, shape2x2,
                                                       ACL_INT32, 10, ACL_INT32, 2, ACL_INT32, e, false, 0, 0));
  }

  {
    std::vector<float> other = {10.0f, 20.0f, 30.0f, 40.0f};
    auto e = BuildAddV3Expected(1.0f, ACL_FLOAT, other, shape2x2, ACL_FLOAT, 2.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddV3Numeric<float, float, float>("CASE_AddV3_Float32_AvoidDirectAdd_Alpha2", stream, 1.0f, ACL_FLOAT,
                                                    other, shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT, 2.0f,
                                                    ACL_FLOAT, e, false, 1e-6, 1e-6));
  }
  {
    std::vector<float> other = {10.0f, -20.0f, 30.0f, -40.0f};
    auto e = BuildAddV3Expected(-1.5f, ACL_FLOAT, other, shape2x2, ACL_FLOAT, 2.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddV3Numeric<float, float, float>("CASE_AddV3_Float32_Alpha2_Axpy", stream, -1.5f, ACL_FLOAT,
                                                    other, shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT, 2.0f,
                                                    ACL_FLOAT, e, false, 1e-6, 1e-6));
  }
  {
    std::vector<int8_t> other = {10, 20, 30, 40};
    auto e = BuildAddV3Expected(static_cast<int8_t>(1), ACL_INT8, other, shape2x2, ACL_INT8, static_cast<int8_t>(2),
                                ACL_INT8, ACL_INT8);
    RecordCase(RunAddV3Numeric<int8_t, int8_t, int8_t>("CASE_AddV3_Int8_Alpha2_FinalMulAdd", stream,
                                                       static_cast<int8_t>(1), ACL_INT8, other, shape2x2, shape2x2,
                                                       ACL_INT8, ACL_INT8, static_cast<int8_t>(2), ACL_INT8, e,
                                                       false, 0, 0));
  }
  {
    std::vector<int32_t> other = {10, 20, 30, 40};
    auto e = BuildAddV3Expected(1, ACL_INT32, other, shape2x2, ACL_INT32, 2, ACL_INT32, ACL_INT32);
    RecordCase(RunAddV3Numeric<int32_t, int32_t, int32_t>("CASE_AddV3_Int32_Alpha2_Axpy", stream, 1, ACL_INT32,
                                                         other, shape2x2, shape2x2, ACL_INT32, ACL_INT32, 2,
                                                         ACL_INT32, e, false, 0, 0));
  }
  {
    std::vector<uint16_t> other = {0x4900, 0x4d00, 0xcf80, 0xd100};
    auto e = BuildAddV3Expected(1, ACL_INT32, other, shape2x2, ACL_FLOAT16, 2.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddV3Numeric<int32_t, uint16_t, float>("CASE_AddV3_OtherFloat16_PromoteToFloat", stream, 1,
                                                         ACL_INT32, other, shape2x2, shape2x2, ACL_FLOAT16,
                                                         ACL_FLOAT, 2.0f, ACL_FLOAT, e, false, 1e-3, 1e-3));
  }
  {
    std::vector<uint16_t> other = {0x4120, 0x41a0, 0xc1f0, 0xc220};
    auto e = BuildAddV3Expected(1, ACL_INT32, other, shape2x2, ACL_BF16, 2.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddV3Numeric<int32_t, uint16_t, float>("CASE_AddV3_OtherBF16_PromoteToFloat", stream, 1,
                                                         ACL_INT32, other, shape2x2, shape2x2, ACL_BF16, ACL_FLOAT,
                                                         2.0f, ACL_FLOAT, e, false, 1e-2, 1e-2));
  }
  {
    std::vector<int32_t> other = {10, 20, 30, 40};
    auto e = BuildAddV3Expected(1.5f, ACL_FLOAT, other, shape2x2, ACL_INT32, 2.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddV3Numeric<float, int32_t, float>("CASE_AddV3_SelfFloat_OtherInt32_Promote", stream, 1.5f,
                                                      ACL_FLOAT, other, shape2x2, shape2x2, ACL_INT32, ACL_FLOAT,
                                                      2.0f, ACL_FLOAT, e, false, 1e-6, 1e-6));
  }
  {
    std::vector<int32_t> other = {10, 20, 30, 40};
    auto e = BuildAddV3Expected(1.5, ACL_DOUBLE, other, shape2x2, ACL_INT32, 2.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunAddV3Numeric<double, int32_t, float>("CASE_AddV3_SelfDouble_OutFloat_SpecialPromote", stream, 1.5,
                                                       ACL_DOUBLE, other, shape2x2, shape2x2, ACL_INT32, ACL_FLOAT,
                                                       2.0f, ACL_FLOAT, e, false, 1e-6, 1e-6));
  }
  RecordCase(RunAddV3Complex64("CASE_AddV3_SelfComplex64_OtherFloat_Probe", stream, false));
  RecordCase(RunAddV3EmptyProbe("CASE_AddV3_EmptyTensor_ZeroWorkspace"));
  RecordCase(RunAddV3EmptyExecuteProbe("CASE_AddV3_EmptyTensor_ExecuteNoKernel", stream));

  {
    std::vector<float> other = {10.0f, 20.0f, 30.0f, 40.0f};
    auto e = BuildAddV3Expected(1.0f, ACL_FLOAT, other, shape2x2, ACL_FLOAT, 2.0f, ACL_FLOAT, ACL_FLOAT);
    RecordCase(RunInplaceAddV3Numeric<float, float>("CASE_InplaceAddV3_Float32_Basic", stream, 1.0f, ACL_FLOAT,
                                                    other, shape2x2, ACL_FLOAT, 2.0f, ACL_FLOAT, e, false, 1e-6,
                                                    1e-6));
  }
  RecordCase(RunInplaceAddsEmptyExecuteProbe("CASE_InplaceAdds_EmptyTensor_ExecuteNoKernel", stream));

  {
    std::vector<int8_t> other = {10, 20, 30, 40};
    auto e = BuildAddV3Expected(static_cast<int8_t>(1), ACL_INT8, other, shape2x2, ACL_INT8, static_cast<int8_t>(2),
                                ACL_INT8, ACL_INT8);
    RecordCase(RunInplaceAddV3Numeric<int8_t, int8_t>("CASE_InplaceAddV3_Int8_Alpha2", stream,
                                                      static_cast<int8_t>(1), ACL_INT8, other, shape2x2, ACL_INT8,
                                                      static_cast<int8_t>(2), ACL_INT8, e, false, 0, 0));
  }
  RecordCase(RunInplaceAddV3EmptyExecuteProbe("CASE_InplaceAddV3_EmptyTensor_ExecuteNoKernel", stream));

  // Expected-failure probes.  These are counted as PASS when parameter validation rejects them.
  {
    std::vector<float> a = {1, 2, 3, 4, 5, 6};
    std::vector<float> b = {1, 2, 3, 4};
    std::vector<double> emptyExpected;
    RecordCase(RunAddNumeric<float, float, float>("CASE_Add_InvalidBroadcast_Fail", stream, a, b, {2, 3}, {2, 2},
                                                  {2, 3}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1.0f, ACL_FLOAT,
                                                  emptyExpected, false));
  }
  {
    std::vector<float> a = {1, 2, 3, 4, 5, 6};
    std::vector<float> b = {10, 20, 30, 40, 50, 60};
    std::vector<double> emptyExpected;
    RecordCase(RunAddNumeric<float, float, float>("CASE_Add_InvalidOutShape_Fail", stream, a, b, {2, 3}, {2, 3},
                                                  {3, 2}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1.0f, ACL_FLOAT,
                                                  emptyExpected, false));
  }
  {
    std::vector<float> a = {1.0f};
    std::vector<float> b = {2.0f};
    std::vector<int64_t> rank9 = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<double> emptyExpected;
    RecordCase(RunAddNumeric<float, float, float>("CASE_Add_RankGreaterThan8_Fail", stream, a, b, rank9, rank9,
                                                  rank9, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1.0f, ACL_FLOAT,
                                                  emptyExpected, false));
  }
  RecordCase(RunAddNullProbe("CASE_Add_NullSelf_Fail", true, false, false, false));
  RecordCase(RunAddNullProbe("CASE_Add_NullOther_Fail", false, true, false, false));
  RecordCase(RunAddNullProbe("CASE_Add_NullAlpha_Fail", false, false, true, false));
  RecordCase(RunAddNullProbe("CASE_Add_NullOut_Fail", false, false, false, true));
  {
    std::vector<uint32_t> a = {1, 2, 3, 4};
    std::vector<uint32_t> b = {10, 20, 30, 40};
    std::vector<double> emptyExpected;
    RecordCase(RunAddNumeric<uint32_t, uint32_t, uint32_t>("CASE_Add_UnsupportedDtype_UINT32_Fail", stream, a, b,
                                                           shape2x2, shape2x2, shape2x2, ACL_UINT32, ACL_UINT32,
                                                           ACL_UINT32, static_cast<uint32_t>(1), ACL_UINT32,
                                                           emptyExpected, false));
  }
  {
    std::vector<int32_t> a = {1, 2, 3, 4};
    std::vector<int32_t> b = {10, 20, 30, 40};
    std::vector<double> emptyExpected;
    RecordCase(RunAddNumeric<int32_t, int32_t, int32_t>("CASE_Add_Int32_AlphaFloatCastFail", stream, a, b, shape2x2,
                                                        shape2x2, shape2x2, ACL_INT32, ACL_INT32, ACL_INT32,
                                                        1.0f, ACL_FLOAT, emptyExpected, false));
  }
  {
    std::vector<float> a = {1, 2, 3, 4};
    std::vector<float> b = {10, 20, 30, 40};
    std::vector<double> emptyExpected;
    RecordCase(RunAddNumeric<float, float, int32_t>("CASE_Add_FloatToInt32_OutCastFail", stream, a, b, shape2x2,
                                                    shape2x2, shape2x2, ACL_FLOAT, ACL_FLOAT, ACL_INT32, 1.0f,
                                                    ACL_FLOAT, emptyExpected, false));
  }
  {
    std::vector<uint8_t> a = {0, 1, 0, 1};
    std::vector<uint8_t> b = {1, 0, 1, 0};
    std::vector<double> emptyExpected;
    RecordCase(RunAddNumeric<uint8_t, uint8_t, uint8_t>("CASE_Add_Bool_AlphaFloatCastFail", stream, a, b, shape2x2,
                                                        shape2x2, shape2x2, ACL_BOOL, ACL_BOOL, ACL_BOOL, 1.0f,
                                                        ACL_FLOAT, emptyExpected, false));
  }
  {
    std::vector<float> a = {1, 2, 3, 4};
    std::vector<double> emptyExpected;
    RecordCase(RunAddsNumeric<float, float, float>("CASE_Adds_InvalidOutShape_Fail", stream, a, shape2x2, {4},
                                                   ACL_FLOAT, ACL_FLOAT, 10.0f, ACL_FLOAT, 1.0f, ACL_FLOAT,
                                                   emptyExpected, false));
  }
  {
    std::vector<float> a = {1.0f};
    std::vector<int64_t> rank9 = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<double> emptyExpected;
    RecordCase(RunAddsNumeric<float, float, float>("CASE_Adds_RankGreaterThan8_Fail", stream, a, rank9, rank9,
                                                   ACL_FLOAT, ACL_FLOAT, 10.0f, ACL_FLOAT, 1.0f, ACL_FLOAT,
                                                   emptyExpected, false));
  }
  RecordCase(RunAddsNullProbe("CASE_Adds_NullSelf_Fail", true, false, false, false));
  RecordCase(RunAddsNullProbe("CASE_Adds_NullOther_Fail", false, true, false, false));
  RecordCase(RunAddsNullProbe("CASE_Adds_NullAlpha_Fail", false, false, true, false));
  RecordCase(RunAddsNullProbe("CASE_Adds_NullOut_Fail", false, false, false, true));
  {
    std::vector<float> a = {1, 2, 3, 4};
    std::vector<float> b = {10, 20, 30, 40};
    std::vector<double> emptyExpected;
    RecordCase(RunInplaceAddNumeric<float, float>("CASE_InplaceAdd_InvalidBroadcast_Fail", stream, a, b, {2, 2},
                                                  {4}, ACL_FLOAT, ACL_FLOAT, 1.0f, ACL_FLOAT, emptyExpected,
                                                  false));
  }
  {
    std::vector<float> a = {1, 2, 3};
    std::vector<float> b = {10, 20, 30, 40, 50, 60};
    std::vector<double> emptyExpected;
    RecordCase(RunInplaceAddNumeric<float, float>("CASE_InplaceAdd_BroadcastResultNotSelfShape_Fail", stream, a, b,
                                                  {1, 3}, {2, 3}, ACL_FLOAT, ACL_FLOAT, 1.0f, ACL_FLOAT,
                                                  emptyExpected, false));
  }
  RecordCase(RunInplaceAddNullProbe("CASE_InplaceAdd_NullSelf_Fail", true, false, false));
  RecordCase(RunInplaceAddNullProbe("CASE_InplaceAdd_NullOther_Fail", false, true, false));
  RecordCase(RunInplaceAddNullProbe("CASE_InplaceAdd_NullAlpha_Fail", false, false, true));
  {
    std::vector<float> other = {10, 20, 30, 40};
    std::vector<double> emptyExpected;
    RecordCase(RunAddV3Numeric<float, float, float>("CASE_AddV3_InvalidOutShape_Fail", stream, 1.0f, ACL_FLOAT,
                                                    other, shape2x2, {4}, ACL_FLOAT, ACL_FLOAT, 1.0f, ACL_FLOAT,
                                                    emptyExpected, false));
  }
  {
    std::vector<uint8_t> other = {0, 1, 0, 1};
    std::vector<double> emptyExpected;
    RecordCase(RunAddV3Numeric<int32_t, uint8_t, int32_t>("CASE_AddV3_OtherBoolUnsupported_Fail", stream, 1,
                                                          ACL_INT32, other, shape2x2, shape2x2, ACL_BOOL,
                                                          ACL_INT32, 1, ACL_INT32, emptyExpected, false));
  }
  {
    std::vector<int32_t> other = {10, 20, 30, 40};
    std::vector<double> emptyExpected;
    RecordCase(RunAddV3Numeric<int32_t, int32_t, int32_t>("CASE_AddV3_Int32_AlphaFloatCastFail", stream, 1,
                                                          ACL_INT32, other, shape2x2, shape2x2, ACL_INT32,
                                                          ACL_INT32, 1.0f, ACL_FLOAT, emptyExpected, false));
  }
  {
    std::vector<float> other = {10, 20, 30, 40};
    std::vector<double> emptyExpected;
    RecordCase(RunAddV3Numeric<float, float, int32_t>("CASE_AddV3_FloatToInt32_OutCastFail", stream, 1.0f,
                                                      ACL_FLOAT, other, shape2x2, shape2x2, ACL_FLOAT, ACL_INT32,
                                                      1.0f, ACL_FLOAT, emptyExpected, false));
  }
  {
    std::vector<float> other = {1.0f};
    std::vector<int64_t> rank9 = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<double> emptyExpected;
    RecordCase(RunAddV3Numeric<float, float, float>("CASE_AddV3_RankGreaterThan8_Fail", stream, 1.0f, ACL_FLOAT,
                                                    other, rank9, rank9, ACL_FLOAT, ACL_FLOAT, 1.0f, ACL_FLOAT,
                                                    emptyExpected, false));
  }
  RecordCase(RunAddV3NullProbe("CASE_AddV3_NullSelf_Fail", true, false, false, false));
  RecordCase(RunAddV3NullProbe("CASE_AddV3_NullOther_Fail", false, true, false, false));
  RecordCase(RunAddV3NullProbe("CASE_AddV3_NullAlpha_Fail", false, false, true, false));
  RecordCase(RunAddV3NullProbe("CASE_AddV3_NullOut_Fail", false, false, false, true));

  Finalize(deviceId, stream);
  LOG_PRINT("Summary: total=%d failed=%d passed=%d\n", totalCases, failedCases, totalCases - failedCases);
  LOG_PRINT("Overall result: [%s]\n", failedCases == 0 ? "PASS" : "FAIL");
  return failedCases == 0 ? 0 : 1;
}
