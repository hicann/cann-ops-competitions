#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

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

namespace {

template <typename T>
struct IsStdComplex : std::false_type {};

template <typename T>
struct IsStdComplex<std::complex<T>> : std::true_type {};

struct TensorResource {
  void* deviceAddr = nullptr;
  aclTensor* tensor = nullptr;
};

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

void DestroyTensorResource(TensorResource& resource) {
  if (resource.tensor != nullptr) {
    aclDestroyTensor(resource.tensor);
    resource.tensor = nullptr;
  }
  if (resource.deviceAddr != nullptr) {
    aclrtFree(resource.deviceAddr);
    resource.deviceAddr = nullptr;
  }
}

void DestroyScalar(aclScalar*& scalar) {
  if (scalar != nullptr) {
    aclDestroyScalar(scalar);
    scalar = nullptr;
  }
}

void FreeWorkspace(void*& workspace) {
  if (workspace != nullptr) {
    aclrtFree(workspace);
    workspace = nullptr;
  }
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData,
                    const std::vector<int64_t>& shape,
                    aclDataType dataType,
                    TensorResource* resource) {
  auto size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
  auto ret = ACL_SUCCESS;
  if (size > 0) {
    ret = aclrtMalloc(&resource->deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(resource->deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  }

  auto strides = MakeStrides(shape);
  resource->tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                                     shape.data(), shape.size(), resource->deviceAddr);
  CHECK_RET(resource->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return 1;);
  return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorWithLayout(const std::vector<T>& hostData,
                              const std::vector<int64_t>& viewShape,
                              const std::vector<int64_t>& storageShape,
                              const std::vector<int64_t>& strides,
                              int64_t offset,
                              aclDataType dataType,
                              TensorResource* resource) {
  auto size = static_cast<size_t>(GetShapeSize(storageShape)) * sizeof(T);
  auto ret = ACL_SUCCESS;
  if (size > 0) {
    ret = aclrtMalloc(&resource->deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(resource->deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  }
  resource->tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, strides.data(), offset, ACL_FORMAT_ND,
                                     storageShape.data(), storageShape.size(), resource->deviceAddr);
  CHECK_RET(resource->tensor != nullptr, LOG_PRINT("aclCreateTensor with layout failed.\n"); return 1;);
  return ACL_SUCCESS;
}

template <typename T>
int CopyDeviceToHost(const TensorResource& resource, const std::vector<int64_t>& shape, std::vector<T>* hostData) {
  hostData->assign(static_cast<size_t>(GetShapeSize(shape)), T{});
  auto size = hostData->size() * sizeof(T);
  if (size == 0) {
    return ACL_SUCCESS;
  }
  auto ret = aclrtMemcpy(hostData->data(), size, resource.deviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

template <typename T>
bool AlmostEqual(T actual, T expected) {
  if constexpr (IsStdComplex<T>::value) {
    return AlmostEqual(actual.real(), expected.real()) && AlmostEqual(actual.imag(), expected.imag());
  } else if constexpr (std::is_floating_point_v<T>) {
    if (std::isnan(actual) && std::isnan(expected)) {
      return true;
    }
    if (std::isinf(actual) && std::isinf(expected)) {
      return std::signbit(actual) == std::signbit(expected);
    }
    const long double atol = std::is_same_v<T, double> ? 1e-9L : 1e-5L;
    const long double rtol = std::is_same_v<T, double> ? 1e-9L : 1e-5L;
    const long double diff = std::fabs(static_cast<long double>(actual) - static_cast<long double>(expected));
    const long double limit = atol + rtol * std::fabs(static_cast<long double>(expected));
    return diff <= limit;
  }
  return actual == expected;
}

template <typename T>
bool CheckVectorEqual(const std::vector<T>& actual, const std::vector<T>& expected, const char* caseName) {
  if (actual.size() != expected.size()) {
    LOG_PRINT("[FAIL] %s size mismatch. actual=%zu expected=%zu\n", caseName, actual.size(), expected.size());
    return false;
  }

  for (size_t i = 0; i < actual.size(); ++i) {
    if (!AlmostEqual(actual[i], expected[i])) {
      if constexpr (IsStdComplex<T>::value) {
        LOG_PRINT("[FAIL] %s mismatch at %zu. actual=(%0.6f,%0.6f) expected=(%0.6f,%0.6f)\n", caseName, i,
                  static_cast<double>(actual[i].real()), static_cast<double>(actual[i].imag()),
                  static_cast<double>(expected[i].real()), static_cast<double>(expected[i].imag()));
      } else if constexpr (std::is_floating_point_v<T>) {
        LOG_PRINT("[FAIL] %s mismatch at %zu. actual=%0.10f expected=%0.10f\n", caseName, i,
                  static_cast<double>(actual[i]), static_cast<double>(expected[i]));
      } else {
        LOG_PRINT("[FAIL] %s mismatch at %zu. actual=%lld expected=%lld\n", caseName, i,
                  static_cast<long long>(actual[i]), static_cast<long long>(expected[i]));
      }
      return false;
    }
  }

  return true;
}

std::vector<int64_t> InferBroadcastShape(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  const size_t rank = a.size() > b.size() ? a.size() : b.size();
  std::vector<int64_t> out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t aDim = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
    const int64_t bDim = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
    if (aDim != bDim && aDim != 1 && bDim != 1) {
      return {};
    }
    out[i] = aDim > bDim ? aDim : bDim;
  }
  return out;
}

std::vector<int64_t> LinearIndexToCoords(int64_t index, const std::vector<int64_t>& shape) {
  std::vector<int64_t> coords(shape.size(), 0);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    coords[static_cast<size_t>(i)] = index % shape[static_cast<size_t>(i)];
    index /= shape[static_cast<size_t>(i)];
  }
  return coords;
}

int64_t BroadcastOffset(const std::vector<int64_t>& coords, const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 0;
  }
  auto strides = MakeStrides(shape);
  const size_t rankDiff = coords.size() - shape.size();
  int64_t offset = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    const int64_t coord = shape[i] == 1 ? 0 : coords[i + rankDiff];
    offset += coord * strides[i];
  }
  return offset;
}

template <typename TSelf, typename TOther, typename TOut>
std::vector<TOut> ComputeMulExpected(const std::vector<TSelf>& selfData,
                                     const std::vector<int64_t>& selfShape,
                                     const std::vector<TOther>& otherData,
                                     const std::vector<int64_t>& otherShape) {
  auto outShape = InferBroadcastShape(selfShape, otherShape);
  std::vector<TOut> expected(static_cast<size_t>(GetShapeSize(outShape)), TOut{});
  for (int64_t i = 0; i < GetShapeSize(outShape); ++i) {
    const auto coords = LinearIndexToCoords(i, outShape);
    const auto selfOffset = BroadcastOffset(coords, selfShape);
    const auto otherOffset = BroadcastOffset(coords, otherShape);
    const long double value = static_cast<long double>(selfData[static_cast<size_t>(selfOffset)]) *
                              static_cast<long double>(otherData[static_cast<size_t>(otherOffset)]);
    expected[static_cast<size_t>(i)] = static_cast<TOut>(value);
  }
  return expected;
}

template <typename T>
float BFloat16BitsToFloat(T bits) {
  uint32_t raw = static_cast<uint32_t>(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

float Float16BitsToFloat(uint16_t bits) {
  const uint32_t sign = (static_cast<uint32_t>(bits & 0x8000U)) << 16;
  const uint32_t exp = (bits >> 10) & 0x1FU;
  const uint32_t mant = bits & 0x03FFU;
  uint32_t raw = 0;
  if (exp == 0) {
    if (mant == 0) {
      raw = sign;
    } else {
      int32_t e = -14;
      uint32_t m = mant;
      while ((m & 0x0400U) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x03FFU;
      raw = sign | (static_cast<uint32_t>(e + 127) << 23) | (m << 13);
    }
  } else if (exp == 0x1FU) {
    raw = sign | 0x7F800000U | (mant << 13);
  } else {
    raw = sign | ((exp + 112U) << 23) | (mant << 13);
  }
  float value = 0.0f;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

uint16_t FloatToBFloat16Bits(float value) {
  uint32_t raw = 0;
  std::memcpy(&raw, &value, sizeof(raw));
  return static_cast<uint16_t>(raw >> 16);
}

uint16_t FloatToFloat16Bits(float value) {
  uint32_t raw = 0;
  std::memcpy(&raw, &value, sizeof(raw));
  const uint32_t sign = (raw >> 16) & 0x8000U;
  int32_t exp = static_cast<int32_t>((raw >> 23) & 0xFFU) - 127 + 15;
  uint32_t mant = raw & 0x7FFFFFU;
  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    mant = (mant | 0x800000U) >> (1 - exp);
    return static_cast<uint16_t>(sign | ((mant + 0x1000U) >> 13));
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00U);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000U) >> 13));
}

uint32_t PackComplex32(float real, float imag) {
  return static_cast<uint32_t>(FloatToFloat16Bits(real)) | (static_cast<uint32_t>(FloatToFloat16Bits(imag)) << 16);
}

std::vector<uint16_t> ComputeFloat16MulExpected(const std::vector<uint16_t>& selfData,
                                                const std::vector<int64_t>& selfShape,
                                                const std::vector<uint16_t>& otherData,
                                                const std::vector<int64_t>& otherShape) {
  auto outShape = InferBroadcastShape(selfShape, otherShape);
  std::vector<uint16_t> expected(static_cast<size_t>(GetShapeSize(outShape)), 0U);
  for (int64_t i = 0; i < GetShapeSize(outShape); ++i) {
    const auto coords = LinearIndexToCoords(i, outShape);
    const auto selfOffset = BroadcastOffset(coords, selfShape);
    const auto otherOffset = BroadcastOffset(coords, otherShape);
    const float value = Float16BitsToFloat(selfData[static_cast<size_t>(selfOffset)]) *
                        Float16BitsToFloat(otherData[static_cast<size_t>(otherOffset)]);
    expected[static_cast<size_t>(i)] = FloatToFloat16Bits(value);
  }
  return expected;
}

std::vector<uint16_t> ComputeBFloat16MulExpected(const std::vector<uint16_t>& selfData,
                                                 const std::vector<int64_t>& selfShape,
                                                 const std::vector<uint16_t>& otherData,
                                                 const std::vector<int64_t>& otherShape) {
  auto outShape = InferBroadcastShape(selfShape, otherShape);
  std::vector<uint16_t> expected(static_cast<size_t>(GetShapeSize(outShape)), 0U);
  for (int64_t i = 0; i < GetShapeSize(outShape); ++i) {
    const auto coords = LinearIndexToCoords(i, outShape);
    const auto selfOffset = BroadcastOffset(coords, selfShape);
    const auto otherOffset = BroadcastOffset(coords, otherShape);
    const float value = BFloat16BitsToFloat(selfData[static_cast<size_t>(selfOffset)]) *
                        BFloat16BitsToFloat(otherData[static_cast<size_t>(otherOffset)]);
    expected[static_cast<size_t>(i)] = FloatToBFloat16Bits(value);
  }
  return expected;
}

std::vector<float> ComputeFloat16FloatMulExpected(const std::vector<uint16_t>& selfData,
                                                  const std::vector<int64_t>& selfShape,
                                                  const std::vector<float>& otherData,
                                                  const std::vector<int64_t>& otherShape) {
  auto outShape = InferBroadcastShape(selfShape, otherShape);
  std::vector<float> expected(static_cast<size_t>(GetShapeSize(outShape)), 0.0f);
  for (int64_t i = 0; i < GetShapeSize(outShape); ++i) {
    const auto coords = LinearIndexToCoords(i, outShape);
    const auto selfOffset = BroadcastOffset(coords, selfShape);
    const auto otherOffset = BroadcastOffset(coords, otherShape);
    expected[static_cast<size_t>(i)] = Float16BitsToFloat(selfData[static_cast<size_t>(selfOffset)]) *
                                       otherData[static_cast<size_t>(otherOffset)];
  }
  return expected;
}

std::vector<float> ComputeFloatBFloat16MulExpected(const std::vector<float>& selfData,
                                                   const std::vector<int64_t>& selfShape,
                                                   const std::vector<uint16_t>& otherData,
                                                   const std::vector<int64_t>& otherShape) {
  auto outShape = InferBroadcastShape(selfShape, otherShape);
  std::vector<float> expected(static_cast<size_t>(GetShapeSize(outShape)), 0.0f);
  for (int64_t i = 0; i < GetShapeSize(outShape); ++i) {
    const auto coords = LinearIndexToCoords(i, outShape);
    const auto selfOffset = BroadcastOffset(coords, selfShape);
    const auto otherOffset = BroadcastOffset(coords, otherShape);
    expected[static_cast<size_t>(i)] = selfData[static_cast<size_t>(selfOffset)] *
                                       BFloat16BitsToFloat(otherData[static_cast<size_t>(otherOffset)]);
  }
  return expected;
}

std::vector<float> ComputeBFloat16FloatMulExpected(const std::vector<uint16_t>& selfData,
                                                   const std::vector<int64_t>& selfShape,
                                                   const std::vector<float>& otherData,
                                                   const std::vector<int64_t>& otherShape) {
  auto outShape = InferBroadcastShape(selfShape, otherShape);
  std::vector<float> expected(static_cast<size_t>(GetShapeSize(outShape)), 0.0f);
  for (int64_t i = 0; i < GetShapeSize(outShape); ++i) {
    const auto coords = LinearIndexToCoords(i, outShape);
    const auto selfOffset = BroadcastOffset(coords, selfShape);
    const auto otherOffset = BroadcastOffset(coords, otherShape);
    expected[static_cast<size_t>(i)] = BFloat16BitsToFloat(selfData[static_cast<size_t>(selfOffset)]) *
                                       otherData[static_cast<size_t>(otherOffset)];
  }
  return expected;
}

template <typename T>
std::vector<T> ComputeMulExpectedDirect(const std::vector<T>& selfData,
                                        const std::vector<int64_t>& selfShape,
                                        const std::vector<T>& otherData,
                                        const std::vector<int64_t>& otherShape) {
  auto outShape = InferBroadcastShape(selfShape, otherShape);
  std::vector<T> expected(static_cast<size_t>(GetShapeSize(outShape)), T{});
  for (int64_t i = 0; i < GetShapeSize(outShape); ++i) {
    const auto coords = LinearIndexToCoords(i, outShape);
    const auto selfOffset = BroadcastOffset(coords, selfShape);
    const auto otherOffset = BroadcastOffset(coords, otherShape);
    expected[static_cast<size_t>(i)] = selfData[static_cast<size_t>(selfOffset)] * otherData[static_cast<size_t>(otherOffset)];
  }
  return expected;
}

template <typename TSelf, typename TOut, typename TScalar>
std::vector<TOut> ComputeMulsExpected(const std::vector<TSelf>& selfData, TScalar scalar) {
  std::vector<TOut> expected(selfData.size(), TOut{});
  for (size_t i = 0; i < selfData.size(); ++i) {
    expected[i] = static_cast<TOut>(static_cast<long double>(selfData[i]) * static_cast<long double>(scalar));
  }
  return expected;
}

bool ExpectFailure(const char* caseName, aclnnStatus actual) {
  if (actual == ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s expected failure but got success.\n", caseName);
    return false;
  }
  LOG_PRINT("[PASS] %s\n", caseName);
  return true;
}

template <typename TSelf, typename TOther, typename TOut>
int RunMulWorkspaceOnlyCase(const char* caseName,
                            const std::vector<TSelf>& selfData,
                            const std::vector<int64_t>& selfShape,
                            aclDataType selfType,
                            const std::vector<TOther>& otherData,
                            const std::vector<int64_t>& otherShape,
                            aclDataType otherType,
                            const std::vector<TOut>& outData,
                            const std::vector<int64_t>& outShape,
                            aclDataType outType,
                            bool expectZeroWorkspace) {
  TensorResource self;
  TensorResource other;
  TensorResource out;
  int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  ret = CreateAclTensor(otherData, otherShape, otherType, &other);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    return 1;
  }
  ret = CreateAclTensor(outData, outShape, outType, &out);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return 1;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
  bool ok = status == ACL_SUCCESS;
  if (!ok) {
    LOG_PRINT("[FAIL] %s aclnnMulGetWorkspaceSize failed. ERROR: %d\n", caseName, status);
  } else if (expectZeroWorkspace && workspaceSize != 0) {
    LOG_PRINT("[FAIL] %s expected zero workspace but got %lu\n", caseName, static_cast<unsigned long>(workspaceSize));
    ok = false;
  } else {
    LOG_PRINT("[PASS] %s\n", caseName);
  }

  DestroyTensorResource(self);
  DestroyTensorResource(other);
  DestroyTensorResource(out);
  return ok ? 0 : 1;
}

template <typename TSelf, typename TOther, typename TOut>
int RunMulExpectedFailureCase(const char* caseName,
                              const std::vector<TSelf>& selfData,
                              const std::vector<int64_t>& selfShape,
                              aclDataType selfType,
                              const std::vector<TOther>& otherData,
                              const std::vector<int64_t>& otherShape,
                              aclDataType otherType,
                              const std::vector<TOut>& outData,
                              const std::vector<int64_t>& outShape,
                              aclDataType outType) {
  TensorResource self;
  TensorResource other;
  TensorResource out;
  int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  ret = CreateAclTensor(otherData, otherShape, otherType, &other);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    return 1;
  }
  ret = CreateAclTensor(outData, outShape, outType, &out);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return 1;
  }
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyTensorResource(self);
  DestroyTensorResource(other);
  DestroyTensorResource(out);
  return ok ? 0 : 1;
}

template <typename TSelf, typename TOut, typename TScalar>
int RunMulsWorkspaceOnlyCase(const char* caseName,
                             const std::vector<TSelf>& selfData,
                             const std::vector<int64_t>& selfShape,
                             aclDataType selfType,
                             TScalar scalarValue,
                             aclDataType scalarType,
                             const std::vector<TOut>& outData,
                             aclDataType outType,
                             bool expectZeroWorkspace) {
  TensorResource self;
  TensorResource out;
  aclScalar* scalar = nullptr;
  int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  ret = CreateAclTensor(outData, selfShape, outType, &out);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return 1;
  }
  scalar = aclCreateScalar(&scalarValue, scalarType);
  CHECK_RET(scalar != nullptr, LOG_PRINT("[FAIL] %s aclCreateScalar failed.\n", caseName);
            DestroyTensorResource(self); DestroyTensorResource(out); return 1;);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulsGetWorkspaceSize(self.tensor, scalar, out.tensor, &workspaceSize, &executor);
  bool ok = status == ACL_SUCCESS;
  if (!ok) {
    LOG_PRINT("[FAIL] %s aclnnMulsGetWorkspaceSize failed. ERROR: %d\n", caseName, status);
  } else if (expectZeroWorkspace && workspaceSize != 0) {
    LOG_PRINT("[FAIL] %s expected zero workspace but got %lu\n", caseName, static_cast<unsigned long>(workspaceSize));
    ok = false;
  } else {
    LOG_PRINT("[PASS] %s\n", caseName);
  }

  DestroyScalar(scalar);
  DestroyTensorResource(self);
  DestroyTensorResource(out);
  return ok ? 0 : 1;
}

template <typename TSelf, typename TOut, typename TScalar>
int RunMulsExpectedFailureCase(const char* caseName,
                               const std::vector<TSelf>& selfData,
                               const std::vector<int64_t>& selfShape,
                               aclDataType selfType,
                               TScalar scalarValue,
                               aclDataType scalarType,
                               const std::vector<TOut>& outData,
                               aclDataType outType) {
  TensorResource self;
  TensorResource out;
  aclScalar* scalar = nullptr;
  int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  ret = CreateAclTensor(outData, selfShape, outType, &out);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return 1;
  }
  scalar = aclCreateScalar(&scalarValue, scalarType);
  CHECK_RET(scalar != nullptr, LOG_PRINT("[FAIL] %s aclCreateScalar failed.\n", caseName);
            DestroyTensorResource(self); DestroyTensorResource(out); return 1;);
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulsGetWorkspaceSize(self.tensor, scalar, out.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyScalar(scalar);
  DestroyTensorResource(self);
  DestroyTensorResource(out);
  return ok ? 0 : 1;
}

template <typename TSelf, typename TOther>
int RunInplaceMulWorkspaceOnlyCase(const char* caseName,
                                   const std::vector<TSelf>& selfData,
                                   const std::vector<int64_t>& selfShape,
                                   aclDataType selfType,
                                   const std::vector<TOther>& otherData,
                                   const std::vector<int64_t>& otherShape,
                                   aclDataType otherType,
                                   bool expectZeroWorkspace) {
  TensorResource self;
  TensorResource other;
  int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  ret = CreateAclTensor(otherData, otherShape, otherType, &other);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    return 1;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnInplaceMulGetWorkspaceSize(self.tensor, other.tensor, &workspaceSize, &executor);
  bool ok = status == ACL_SUCCESS;
  if (!ok) {
    LOG_PRINT("[FAIL] %s aclnnInplaceMulGetWorkspaceSize failed. ERROR: %d\n", caseName, status);
  } else if (expectZeroWorkspace && workspaceSize != 0) {
    LOG_PRINT("[FAIL] %s expected zero workspace but got %lu\n", caseName, static_cast<unsigned long>(workspaceSize));
    ok = false;
  } else {
    LOG_PRINT("[PASS] %s\n", caseName);
  }

  DestroyTensorResource(self);
  DestroyTensorResource(other);
  return ok ? 0 : 1;
}

template <typename TSelf, typename TOther>
int RunInplaceMulExpectedFailureCase(const char* caseName,
                                     const std::vector<TSelf>& selfData,
                                     const std::vector<int64_t>& selfShape,
                                     aclDataType selfType,
                                     const std::vector<TOther>& otherData,
                                     const std::vector<int64_t>& otherShape,
                                     aclDataType otherType) {
  TensorResource self;
  TensorResource other;
  int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  ret = CreateAclTensor(otherData, otherShape, otherType, &other);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    return 1;
  }
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnInplaceMulGetWorkspaceSize(self.tensor, other.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyTensorResource(self);
  DestroyTensorResource(other);
  return ok ? 0 : 1;
}

template <typename TSelf, typename TScalar>
int RunInplaceMulsWorkspaceOnlyCase(const char* caseName,
                                    const std::vector<TSelf>& selfData,
                                    const std::vector<int64_t>& selfShape,
                                    aclDataType selfType,
                                    TScalar scalarValue,
                                    aclDataType scalarType,
                                    bool expectZeroWorkspace) {
  TensorResource self;
  aclScalar* scalar = nullptr;
  int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  scalar = aclCreateScalar(&scalarValue, scalarType);
  CHECK_RET(scalar != nullptr, LOG_PRINT("[FAIL] %s aclCreateScalar failed.\n", caseName);
            DestroyTensorResource(self); return 1;);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnInplaceMulsGetWorkspaceSize(self.tensor, scalar, &workspaceSize, &executor);
  bool ok = status == ACL_SUCCESS;
  if (!ok) {
    LOG_PRINT("[FAIL] %s aclnnInplaceMulsGetWorkspaceSize failed. ERROR: %d\n", caseName, status);
  } else if (expectZeroWorkspace && workspaceSize != 0) {
    LOG_PRINT("[FAIL] %s expected zero workspace but got %lu\n", caseName, static_cast<unsigned long>(workspaceSize));
    ok = false;
  } else {
    LOG_PRINT("[PASS] %s\n", caseName);
  }

  DestroyScalar(scalar);
  DestroyTensorResource(self);
  return ok ? 0 : 1;
}

template <typename TSelf, typename TScalar>
int RunInplaceMulsExpectedFailureCase(const char* caseName,
                                      const std::vector<TSelf>& selfData,
                                      const std::vector<int64_t>& selfShape,
                                      aclDataType selfType,
                                      TScalar scalarValue,
                                      aclDataType scalarType) {
  TensorResource self;
  aclScalar* scalar = nullptr;
  int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  scalar = aclCreateScalar(&scalarValue, scalarType);
  CHECK_RET(scalar != nullptr, LOG_PRINT("[FAIL] %s aclCreateScalar failed.\n", caseName);
            DestroyTensorResource(self); return 1;);
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnInplaceMulsGetWorkspaceSize(self.tensor, scalar, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyScalar(scalar);
  DestroyTensorResource(self);
  return ok ? 0 : 1;
}

template <typename TSelf, typename TOther, typename TOut>
int RunMulCase(const char* caseName,
               const std::vector<TSelf>& selfData,
               const std::vector<int64_t>& selfShape,
               aclDataType selfType,
               const std::vector<TOther>& otherData,
               const std::vector<int64_t>& otherShape,
               aclDataType otherType,
               const std::vector<TOut>& expected,
               const std::vector<int64_t>& outShape,
               aclDataType outType,
               aclrtStream stream) {
  TensorResource self;
  TensorResource other;
  TensorResource out;
  void* workspaceAddr = nullptr;
  int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  ret = CreateAclTensor(otherData, otherShape, otherType, &other);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    return 1;
  }
  std::vector<TOut> outInit(static_cast<size_t>(GetShapeSize(outShape)), TOut{});
  ret = CreateAclTensor(outInit, outShape, outType, &out);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return 1;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto status = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
  if (status != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclnnMulGetWorkspaceSize failed. ERROR: %d\n", caseName, status);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return 1;
  }

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[FAIL] %s allocate workspace failed. ERROR: %d\n", caseName, ret);
      DestroyTensorResource(self);
      DestroyTensorResource(other);
      DestroyTensorResource(out);
      return 1;
    }
  }

  status = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  if (status != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclnnMul failed. ERROR: %d\n", caseName, status);
    FreeWorkspace(workspaceAddr);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return 1;
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclrtSynchronizeStream failed. ERROR: %d\n", caseName, ret);
    FreeWorkspace(workspaceAddr);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return 1;
  }

  std::vector<TOut> actual;
  ret = CopyDeviceToHost(out, outShape, &actual);
  const bool ok = (ret == ACL_SUCCESS) && CheckVectorEqual(actual, expected, caseName);
  if (ok) {
    LOG_PRINT("[PASS] %s\n", caseName);
  }

  FreeWorkspace(workspaceAddr);
  DestroyTensorResource(self);
  DestroyTensorResource(other);
  DestroyTensorResource(out);
  return ok ? 0 : 1;
}

template <typename TSelf, typename TOut, typename TScalar>
int RunMulsCase(const char* caseName,
                const std::vector<TSelf>& selfData,
                const std::vector<int64_t>& selfShape,
                aclDataType selfType,
                TScalar scalarValue,
                aclDataType scalarType,
                const std::vector<TOut>& expected,
                aclDataType outType,
                aclrtStream stream) {
  TensorResource self;
  TensorResource out;
  aclScalar* scalar = nullptr;
  void* workspaceAddr = nullptr;
  int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }

  std::vector<TOut> outInit(selfData.size(), TOut{});
  ret = CreateAclTensor(outInit, selfShape, outType, &out);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return 1;
  }

  scalar = aclCreateScalar(&scalarValue, scalarType);
  CHECK_RET(scalar != nullptr, LOG_PRINT("[FAIL] %s aclCreateScalar failed.\n", caseName);
            DestroyTensorResource(self); DestroyTensorResource(out); return 1;);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto status = aclnnMulsGetWorkspaceSize(self.tensor, scalar, out.tensor, &workspaceSize, &executor);
  if (status != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclnnMulsGetWorkspaceSize failed. ERROR: %d\n", caseName, status);
    DestroyScalar(scalar);
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return 1;
  }

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[FAIL] %s allocate workspace failed. ERROR: %d\n", caseName, ret);
      DestroyScalar(scalar);
      DestroyTensorResource(self);
      DestroyTensorResource(out);
      return 1;
    }
  }

  status = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  if (status != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclnnMuls failed. ERROR: %d\n", caseName, status);
    FreeWorkspace(workspaceAddr);
    DestroyScalar(scalar);
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return 1;
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclrtSynchronizeStream failed. ERROR: %d\n", caseName, ret);
    FreeWorkspace(workspaceAddr);
    DestroyScalar(scalar);
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return 1;
  }

  std::vector<TOut> actual;
  ret = CopyDeviceToHost(out, selfShape, &actual);
  const bool ok = (ret == ACL_SUCCESS) && CheckVectorEqual(actual, expected, caseName);
  if (ok) {
    LOG_PRINT("[PASS] %s\n", caseName);
  }

  FreeWorkspace(workspaceAddr);
  DestroyScalar(scalar);
  DestroyTensorResource(self);
  DestroyTensorResource(out);
  return ok ? 0 : 1;
}

template <typename TSelf, typename TOther>
int RunInplaceMulCase(const char* caseName,
                      const std::vector<TSelf>& selfData,
                      const std::vector<int64_t>& selfShape,
                      aclDataType selfType,
                      const std::vector<TOther>& otherData,
                      const std::vector<int64_t>& otherShape,
                      aclDataType otherType,
                      const std::vector<TSelf>& expected,
                      aclrtStream stream) {
  TensorResource self;
  TensorResource other;
  void* workspaceAddr = nullptr;
  int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  ret = CreateAclTensor(otherData, otherShape, otherType, &other);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    return 1;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto status = aclnnInplaceMulGetWorkspaceSize(self.tensor, other.tensor, &workspaceSize, &executor);
  if (status != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclnnInplaceMulGetWorkspaceSize failed. ERROR: %d\n", caseName, status);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    return 1;
  }

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[FAIL] %s allocate workspace failed. ERROR: %d\n", caseName, ret);
      DestroyTensorResource(self);
      DestroyTensorResource(other);
      return 1;
    }
  }

  status = aclnnInplaceMul(workspaceAddr, workspaceSize, executor, stream);
  if (status != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclnnInplaceMul failed. ERROR: %d\n", caseName, status);
    FreeWorkspace(workspaceAddr);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    return 1;
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclrtSynchronizeStream failed. ERROR: %d\n", caseName, ret);
    FreeWorkspace(workspaceAddr);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    return 1;
  }

  std::vector<TSelf> actual;
  ret = CopyDeviceToHost(self, selfShape, &actual);
  const bool ok = (ret == ACL_SUCCESS) && CheckVectorEqual(actual, expected, caseName);
  if (ok) {
    LOG_PRINT("[PASS] %s\n", caseName);
  }

  FreeWorkspace(workspaceAddr);
  DestroyTensorResource(self);
  DestroyTensorResource(other);
  return ok ? 0 : 1;
}

template <typename TSelf, typename TScalar>
int RunInplaceMulsCase(const char* caseName,
                       const std::vector<TSelf>& selfData,
                       const std::vector<int64_t>& selfShape,
                       aclDataType selfType,
                       TScalar scalarValue,
                       aclDataType scalarType,
                       const std::vector<TSelf>& expected,
                       aclrtStream stream) {
  TensorResource self;
  aclScalar* scalar = nullptr;
  void* workspaceAddr = nullptr;
  int ret = CreateAclTensor(selfData, selfShape, selfType, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }

  scalar = aclCreateScalar(&scalarValue, scalarType);
  CHECK_RET(scalar != nullptr, LOG_PRINT("[FAIL] %s aclCreateScalar failed.\n", caseName);
            DestroyTensorResource(self); return 1;);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto status = aclnnInplaceMulsGetWorkspaceSize(self.tensor, scalar, &workspaceSize, &executor);
  if (status != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclnnInplaceMulsGetWorkspaceSize failed. ERROR: %d\n", caseName, status);
    DestroyScalar(scalar);
    DestroyTensorResource(self);
    return 1;
  }

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("[FAIL] %s allocate workspace failed. ERROR: %d\n", caseName, ret);
      DestroyScalar(scalar);
      DestroyTensorResource(self);
      return 1;
    }
  }

  status = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, stream);
  if (status != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclnnInplaceMuls failed. ERROR: %d\n", caseName, status);
    FreeWorkspace(workspaceAddr);
    DestroyScalar(scalar);
    DestroyTensorResource(self);
    return 1;
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("[FAIL] %s aclrtSynchronizeStream failed. ERROR: %d\n", caseName, ret);
    FreeWorkspace(workspaceAddr);
    DestroyScalar(scalar);
    DestroyTensorResource(self);
    return 1;
  }

  std::vector<TSelf> actual;
  ret = CopyDeviceToHost(self, selfShape, &actual);
  const bool ok = (ret == ACL_SUCCESS) && CheckVectorEqual(actual, expected, caseName);
  if (ok) {
    LOG_PRINT("[PASS] %s\n", caseName);
  }

  FreeWorkspace(workspaceAddr);
  DestroyScalar(scalar);
  DestroyTensorResource(self);
  return ok ? 0 : 1;
}

int RunNullSelfCase(const char* caseName) {
  TensorResource other;
  TensorResource out;
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> otherData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outData = {0.0f, 0.0f, 0.0f, 0.0f};
  int ret = CreateAclTensor(otherData, shape, ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(other);
    return 1;
  }
  ret = CreateAclTensor(outData, shape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return 1;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulGetWorkspaceSize(nullptr, other.tensor, out.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyTensorResource(other);
  DestroyTensorResource(out);
  return ok ? 0 : 1;
}

int RunNullOtherCase(const char* caseName) {
  TensorResource self;
  TensorResource out;
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  int ret = CreateAclTensor(data, shape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) return 1;
  ret = CreateAclTensor(data, shape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) { DestroyTensorResource(self); return 1; }
  uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulGetWorkspaceSize(self.tensor, nullptr, out.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyTensorResource(self); DestroyTensorResource(out); return ok ? 0 : 1;
}

int RunNullOutCase(const char* caseName) {
  TensorResource self;
  TensorResource other;
  std::vector<int64_t> shape = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  int ret = CreateAclTensor(data, shape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) return 1;
  ret = CreateAclTensor(data, shape, ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) { DestroyTensorResource(self); return 1; }
  uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, nullptr, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyTensorResource(self); DestroyTensorResource(other); return ok ? 0 : 1;
}

int RunInvalidBroadcastCase(const char* caseName) {
  TensorResource self;
  TensorResource other;
  TensorResource out;
  std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> otherData = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> outData(6, 0.0f);
  const std::vector<int64_t> selfShape = {2, 3};
  const std::vector<int64_t> otherShape = {2, 2};
  const std::vector<int64_t> outShape = {2, 3};

  int ret = CreateAclTensor(selfData, selfShape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  ret = CreateAclTensor(otherData, otherShape, ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    return 1;
  }
  ret = CreateAclTensor(outData, outShape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return 1;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyTensorResource(self);
  DestroyTensorResource(other);
  DestroyTensorResource(out);
  return ok ? 0 : 1;
}

int RunUnsupportedDtypeCase(const char* caseName) {
  TensorResource self;
  TensorResource other;
  TensorResource out;
  std::vector<uint32_t> selfData = {1U, 2U, 3U, 4U};
  std::vector<uint32_t> otherData = {5U, 6U, 7U, 8U};
  std::vector<uint32_t> outData(4, 0U);
  const std::vector<int64_t> shape = {2, 2};

  int ret = CreateAclTensor(selfData, shape, ACL_UINT32, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  ret = CreateAclTensor(otherData, shape, ACL_UINT32, &other);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    return 1;
  }
  ret = CreateAclTensor(outData, shape, ACL_UINT32, &out);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return 1;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyTensorResource(self);
  DestroyTensorResource(other);
  DestroyTensorResource(out);
  return ok ? 0 : 1;
}

int RunInplaceInvalidBroadcastCase(const char* caseName) {
  TensorResource self;
  TensorResource other;
  std::vector<float> selfData = {1.0f, 2.0f, 3.0f};
  std::vector<float> otherData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<int64_t> selfShape = {1, 3};
  const std::vector<int64_t> otherShape = {2, 3};

  int ret = CreateAclTensor(selfData, selfShape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  ret = CreateAclTensor(otherData, otherShape, ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    return 1;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnInplaceMulGetWorkspaceSize(self.tensor, other.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyTensorResource(self);
  DestroyTensorResource(other);
  return ok ? 0 : 1;
}

int RunMulsNullScalarCase(const char* caseName) {
  TensorResource self;
  TensorResource out;
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<int64_t> shape = {2, 2};
  int ret = CreateAclTensor(data, shape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  ret = CreateAclTensor(data, shape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return 1;
  }
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulsGetWorkspaceSize(self.tensor, nullptr, out.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyTensorResource(self);
  DestroyTensorResource(out);
  return ok ? 0 : 1;
}

int RunMulsNullSelfCase(const char* caseName) {
  TensorResource out;
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<int64_t> shape = {2, 2};
  int ret = CreateAclTensor(data, shape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) return 1;
  float scalarValue = 2.0f;
  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar != nullptr, DestroyTensorResource(out); return 1;);
  uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulsGetWorkspaceSize(nullptr, scalar, out.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyScalar(scalar); DestroyTensorResource(out); return ok ? 0 : 1;
}

int RunMulsNullOutCase(const char* caseName) {
  TensorResource self;
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<int64_t> shape = {2, 2};
  int ret = CreateAclTensor(data, shape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) return 1;
  float scalarValue = 2.0f;
  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar != nullptr, DestroyTensorResource(self); return 1;);
  uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulsGetWorkspaceSize(self.tensor, scalar, nullptr, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyScalar(scalar); DestroyTensorResource(self); return ok ? 0 : 1;
}

int RunMulsUnsupportedDtypeCase(const char* caseName) {
  TensorResource self;
  TensorResource out;
  const std::vector<uint32_t> data = {1U, 2U, 3U, 4U};
  const std::vector<int64_t> shape = {2, 2};
  int ret = CreateAclTensor(data, shape, ACL_UINT32, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  ret = CreateAclTensor(data, shape, ACL_UINT32, &out);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(out);
    return 1;
  }
  float scalarValue = 2.0f;
  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar != nullptr, DestroyTensorResource(self); DestroyTensorResource(out); return 1;);
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulsGetWorkspaceSize(self.tensor, scalar, out.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyScalar(scalar);
  DestroyTensorResource(self);
  DestroyTensorResource(out);
  return ok ? 0 : 1;
}

int RunInplaceMulsNullScalarCase(const char* caseName) {
  TensorResource self;
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<int64_t> shape = {2, 2};
  int ret = CreateAclTensor(data, shape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnInplaceMulsGetWorkspaceSize(self.tensor, nullptr, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyTensorResource(self);
  return ok ? 0 : 1;
}

int RunInplaceMulNullSelfCase(const char* caseName) {
  TensorResource other;
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<int64_t> shape = {2, 2};
  int ret = CreateAclTensor(data, shape, ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) return 1;
  uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr;
  const auto status = aclnnInplaceMulGetWorkspaceSize(nullptr, other.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyTensorResource(other); return ok ? 0 : 1;
}

int RunInplaceMulNullOtherCase(const char* caseName) {
  TensorResource self;
  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<int64_t> shape = {2, 2};
  int ret = CreateAclTensor(data, shape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) return 1;
  uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr;
  const auto status = aclnnInplaceMulGetWorkspaceSize(self.tensor, nullptr, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyTensorResource(self); return ok ? 0 : 1;
}

int RunInplaceMulsNullSelfCase(const char* caseName) {
  float scalarValue = 2.0f;
  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar != nullptr, return 1;);
  uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr;
  const auto status = aclnnInplaceMulsGetWorkspaceSize(nullptr, scalar, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyScalar(scalar); return ok ? 0 : 1;
}

int RunMulWrongOutShapeCase(const char* caseName) {
  TensorResource self, other, out;
  const std::vector<float> selfData = {1,2,3,4,5,6};
  const std::vector<float> otherData = {10,20,30};
  const std::vector<float> outData(4, 0.0f);
  const std::vector<int64_t> selfShape = {2,3};
  const std::vector<int64_t> otherShape = {3};
  const std::vector<int64_t> outShape = {2,2};
  int ret = CreateAclTensor(selfData, selfShape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) return 1;
  ret = CreateAclTensor(otherData, otherShape, ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) { DestroyTensorResource(self); return 1; }
  ret = CreateAclTensor(outData, outShape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) { DestroyTensorResource(self); DestroyTensorResource(other); return 1; }
  uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyTensorResource(self); DestroyTensorResource(other); DestroyTensorResource(out); return ok ? 0 : 1;
}

int RunMulsWrongOutShapeCase(const char* caseName) {
  TensorResource self, out;
  const std::vector<float> selfData = {1,2,3,4};
  const std::vector<float> outData(3, 0.0f);
  const std::vector<int64_t> selfShape = {2,2};
  const std::vector<int64_t> outShape = {3};
  int ret = CreateAclTensor(selfData, selfShape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) return 1;
  ret = CreateAclTensor(outData, outShape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) { DestroyTensorResource(self); return 1; }
  float scalarValue = 2.0f;
  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar != nullptr, DestroyTensorResource(self); DestroyTensorResource(out); return 1;);
  uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulsGetWorkspaceSize(self.tensor, scalar, out.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyScalar(scalar); DestroyTensorResource(self); DestroyTensorResource(out); return ok ? 0 : 1;
}

int RunMulOverMaxDimCase(const char* caseName) {
  TensorResource self, other, out;
  const std::vector<int64_t> shape(9, 1);
  const std::vector<float> data = {1.0f};
  int ret = CreateAclTensor(data, shape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) return 1;
  ret = CreateAclTensor(data, shape, ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) { DestroyTensorResource(self); return 1; }
  ret = CreateAclTensor(data, shape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) { DestroyTensorResource(self); DestroyTensorResource(other); return 1; }
  uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyTensorResource(self); DestroyTensorResource(other); DestroyTensorResource(out); return ok ? 0 : 1;
}

int RunInplaceMulsUnsupportedDtypeCase(const char* caseName) {
  TensorResource self;
  const std::vector<uint32_t> data = {1U, 2U, 3U, 4U};
  const std::vector<int64_t> shape = {2, 2};
  int ret = CreateAclTensor(data, shape, ACL_UINT32, &self);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  float scalarValue = 2.0f;
  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar != nullptr, DestroyTensorResource(self); return 1;);
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnInplaceMulsGetWorkspaceSize(self.tensor, scalar, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyScalar(scalar);
  DestroyTensorResource(self);
  return ok ? 0 : 1;
}

int RunMulsInvalidOutDtypeCase(const char* caseName) {
  TensorResource self;
  TensorResource out;
  const std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<uint8_t> outData(4, 0);
  const std::vector<int64_t> shape = {2, 2};
  int ret = CreateAclTensor(selfData, shape, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) return 1;
  ret = CreateAclTensor(outData, shape, ACL_BOOL, &out);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  float scalarValue = 2.0f;
  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar != nullptr, DestroyTensorResource(self); DestroyTensorResource(out); return 1;);
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnMulsGetWorkspaceSize(self.tensor, scalar, out.tensor, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyScalar(scalar);
  DestroyTensorResource(self);
  DestroyTensorResource(out);
  return ok ? 0 : 1;
}

int RunInplaceMulsInvalidResultTypeCase(const char* caseName) {
  TensorResource self;
  const std::vector<int32_t> selfData = {1, 2, 3, 4};
  const std::vector<int64_t> shape = {2, 2};
  int ret = CreateAclTensor(selfData, shape, ACL_INT32, &self);
  if (ret != ACL_SUCCESS) return 1;
  float scalarValue = 2.0f;
  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar != nullptr, DestroyTensorResource(self); return 1;);
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnInplaceMulsGetWorkspaceSize(self.tensor, scalar, &workspaceSize, &executor);
  const bool ok = ExpectFailure(caseName, status);
  DestroyScalar(scalar);
  DestroyTensorResource(self);
  return ok ? 0 : 1;
}

int RunNonContiguousMulCase(const char* caseName, aclrtStream stream) {
  TensorResource self;
  TensorResource other;
  TensorResource out;
  const std::vector<float> selfStorage = {1.0f, 2.0f, 99.0f, 3.0f, 4.0f, 99.0f};
  const std::vector<float> otherData = {10.0f, 20.0f, 30.0f, 40.0f};
  const std::vector<float> expected = {10.0f, 40.0f, 90.0f, 160.0f};
  const std::vector<float> outInit(4, 0.0f);
  const std::vector<int64_t> viewShape = {2, 2};
  const std::vector<int64_t> storageShape = {2, 3};
  const std::vector<int64_t> strides = {3, 1};
  int ret = CreateAclTensorWithLayout(selfStorage, viewShape, storageShape, strides, 0, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) return 1;
  ret = CreateAclTensor(otherData, viewShape, ACL_FLOAT, &other);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    return 1;
  }
  ret = CreateAclTensor(outInit, viewShape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    return 1;
  }
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto status = aclnnMulGetWorkspaceSize(self.tensor, other.tensor, out.tensor, &workspaceSize, &executor);
  if (status != ACL_SUCCESS) {
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return 1;
  }
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      DestroyTensorResource(self);
      DestroyTensorResource(other);
      DestroyTensorResource(out);
      return 1;
    }
  }
  status = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
  if (status != ACL_SUCCESS) {
    FreeWorkspace(workspaceAddr);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return 1;
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    FreeWorkspace(workspaceAddr);
    DestroyTensorResource(self);
    DestroyTensorResource(other);
    DestroyTensorResource(out);
    return 1;
  }
  std::vector<float> actual;
  ret = CopyDeviceToHost(out, viewShape, &actual);
  const bool ok = (ret == ACL_SUCCESS) && CheckVectorEqual(actual, expected, caseName);
  if (ok) LOG_PRINT("[PASS] %s\n", caseName);
  FreeWorkspace(workspaceAddr);
  DestroyTensorResource(self);
  DestroyTensorResource(other);
  DestroyTensorResource(out);
  return ok ? 0 : 1;
}

int RunNonContiguousMulsCase(const char* caseName, aclrtStream stream) {
  TensorResource self;
  TensorResource out;
  const std::vector<float> selfStorage = {1.0f, 2.0f, 99.0f, 3.0f, 4.0f, 99.0f};
  const std::vector<float> expected = {2.0f, 4.0f, 6.0f, 8.0f};
  const std::vector<float> outInit(4, 0.0f);
  const std::vector<int64_t> viewShape = {2, 2};
  const std::vector<int64_t> storageShape = {2, 3};
  const std::vector<int64_t> strides = {3, 1};
  int ret = CreateAclTensorWithLayout(selfStorage, viewShape, storageShape, strides, 0, ACL_FLOAT, &self);
  if (ret != ACL_SUCCESS) return 1;
  ret = CreateAclTensor(outInit, viewShape, ACL_FLOAT, &out);
  if (ret != ACL_SUCCESS) { DestroyTensorResource(self); return 1; }
  float scalarValue = 2.0f;
  aclScalar* scalar = aclCreateScalar(&scalarValue, ACL_FLOAT);
  CHECK_RET(scalar != nullptr, DestroyTensorResource(self); DestroyTensorResource(out); return 1;);
  uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr; void* workspaceAddr = nullptr;
  auto status = aclnnMulsGetWorkspaceSize(self.tensor, scalar, out.tensor, &workspaceSize, &executor);
  if (status != ACL_SUCCESS) { DestroyScalar(scalar); DestroyTensorResource(self); DestroyTensorResource(out); return 1; }
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { DestroyScalar(scalar); DestroyTensorResource(self); DestroyTensorResource(out); return 1; }
  }
  status = aclnnMuls(workspaceAddr, workspaceSize, executor, stream);
  if (status != ACL_SUCCESS) { FreeWorkspace(workspaceAddr); DestroyScalar(scalar); DestroyTensorResource(self); DestroyTensorResource(out); return 1; }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) { FreeWorkspace(workspaceAddr); DestroyScalar(scalar); DestroyTensorResource(self); DestroyTensorResource(out); return 1; }
  std::vector<float> actual;
  ret = CopyDeviceToHost(out, viewShape, &actual);
  const bool ok = (ret == ACL_SUCCESS) && CheckVectorEqual(actual, expected, caseName);
  if (ok) LOG_PRINT("[PASS] %s\n", caseName);
  FreeWorkspace(workspaceAddr); DestroyScalar(scalar); DestroyTensorResource(self); DestroyTensorResource(out);
  return ok ? 0 : 1;
}

}  // namespace

int main() {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int failed = 0;

  // Branch family: baseline tensor*tensor execution with explicit result validation.
  {
    const std::vector<float> selfData = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    const std::vector<float> otherData = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f};
    const std::vector<int64_t> shape = {4, 2};
    failed += RunMulCase("mul_float32_basic", selfData, shape, ACL_FLOAT, otherData, shape,
                         ACL_FLOAT, ComputeMulExpected<float, float, float>(selfData, shape, otherData, shape),
                         shape, ACL_FLOAT, stream);
  }

  // Branch family: broadcast success on the legal int32->int32 path.
  {
    const std::vector<int32_t> selfData = {1, 2, 3, 4, 5, 6};
    const std::vector<int32_t> otherData = {10, 20, 30};
    const std::vector<int64_t> selfShape = {2, 3};
    const std::vector<int64_t> otherShape = {3};
    const auto outShape = InferBroadcastShape(selfShape, otherShape);
    failed += RunMulCase("mul_int32_broadcast", selfData, selfShape, ACL_INT32, otherData,
                         otherShape, ACL_INT32,
                         ComputeMulExpected<int32_t, int32_t, int32_t>(selfData, selfShape, otherData, otherShape),
                         outShape, ACL_INT32, stream);
  }

  // Branch family: double route in op_api/mul.cpp.
  {
    const std::vector<double> selfData = {-1.5, 2.0, -3.5, 4.0};
    const std::vector<double> otherData = {2.0, -0.5, -2.0, 0.25};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunMulCase("mul_double_same_shape", selfData, shape, ACL_DOUBLE, otherData, shape,
                         ACL_DOUBLE,
                           ComputeMulExpected<double, double, double>(selfData, shape, otherData, shape), shape,
                           ACL_DOUBLE, stream);
  }

  {
    const std::vector<double> selfData = {std::numeric_limits<double>::infinity(),
                                          -std::numeric_limits<double>::infinity(),
                                          std::numeric_limits<double>::quiet_NaN(), 0.0};
    const std::vector<double> otherData = {2.0, -1.0, 3.0, 5.0};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunMulCase("mul_double_nan_inf_boundary", selfData, shape, ACL_DOUBLE, otherData, shape,
                         ACL_DOUBLE,
                         ComputeMulExpected<double, double, double>(selfData, shape, otherData, shape), shape,
                         ACL_DOUBLE, stream);
  }

  // Branch family: float+double promote path.
  {
    const std::vector<float> selfData = {1.5f, -2.0f, 3.25f, -4.5f};
    const std::vector<double> otherData = {2.0, -0.5, 4.0, 0.25};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<double> outData(4, 0.0);
    failed += RunMulExpectedFailureCase("mul_float_double_promote_out_double", selfData, shape, ACL_FLOAT, otherData,
                                        shape, ACL_DOUBLE, outData, shape, ACL_DOUBLE);
  }

  // Branch family: unsigned integer tiling branch.
  {
    const std::vector<uint8_t> selfData = {1, 2, 3, 4};
    const std::vector<uint8_t> otherData = {5, 6, 7, 8};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunMulCase("mul_uint8_same_shape", selfData, shape, ACL_UINT8, otherData, shape,
                         ACL_UINT8,
                         ComputeMulExpected<uint8_t, uint8_t, uint8_t>(selfData, shape, otherData, shape), shape,
                         ACL_UINT8, stream);
  }

  // Branch family: additional int8 tiling entry with end-to-end validation.
  {
    const std::vector<int8_t> selfData = {1, -2, 3, -4};
    const std::vector<int8_t> otherData = {-1, 2, -3, 4};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<int8_t> expected = ComputeMulExpected<int8_t, int8_t, int8_t>(selfData, shape, otherData, shape);
    failed += RunMulCase("mul_int8_same_shape", selfData, shape, ACL_INT8, otherData, shape,
                         ACL_INT8, expected, shape, ACL_INT8, stream);
  }

  // Branch family: additional int16 tiling entry with end-to-end validation.
  {
    const std::vector<int16_t> selfData = {1, -2, 3, -4};
    const std::vector<int16_t> otherData = {-1, 2, -3, 4};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<int16_t> expected = ComputeMulExpected<int16_t, int16_t, int16_t>(selfData, shape, otherData, shape);
    failed += RunMulCase("mul_int16_same_shape", selfData, shape, ACL_INT16, otherData, shape,
                         ACL_INT16, expected, shape, ACL_INT16, stream);
  }

  // Branch family: additional int64 tiling entry with end-to-end validation.
  {
    const std::vector<int64_t> selfData = {1, -2, 3, -4};
    const std::vector<int64_t> otherData = {-1, 2, -3, 4};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<int64_t> expected = ComputeMulExpected<int64_t, int64_t, int64_t>(selfData, shape, otherData, shape);
    failed += RunMulCase("mul_int64_same_shape", selfData, shape, ACL_INT64, otherData, shape,
                         ACL_INT64, expected, shape, ACL_INT64, stream);
  }

  // Branch family: bool tiling entry with end-to-end validation.
  {
    const std::vector<uint8_t> selfData = {1, 0, 1, 1};
    const std::vector<uint8_t> otherData = {1, 1, 0, 1};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint8_t> expected = ComputeMulExpected<uint8_t, uint8_t, uint8_t>(selfData, shape, otherData, shape);
    failed += RunMulCase("mul_bool_same_shape", selfData, shape, ACL_BOOL, otherData, shape,
                         ACL_BOOL, expected, shape, ACL_BOOL, stream);
  }

  // Branch family: fp16*float mixed path in the reverse order.
  {
    const std::vector<uint16_t> selfData = {0x3c00, 0x4000, 0x4200, 0x4400};
    const std::vector<float> otherData = {1.5f, -2.0f, 3.25f, -4.5f};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> expected = ComputeFloat16FloatMulExpected(selfData, shape, otherData, shape);
    failed += RunMulCase("mul_fp16_float_promote_out_float", selfData, shape, ACL_FLOAT16, otherData, shape,
                         ACL_FLOAT, expected, shape, ACL_FLOAT, stream);
  }

  // Branch family: bf16*float mixed path in the reverse order.
  {
    const std::vector<uint16_t> selfData = {0x3f80, 0x4000, 0x4040, 0x4080};
    const std::vector<float> otherData = {1.5f, -2.0f, 3.25f, -4.5f};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> expected = ComputeBFloat16FloatMulExpected(selfData, shape, otherData, shape);
    failed += RunMulCase("mul_bf16_float_promote_out_float", selfData, shape, ACL_BF16, otherData, shape,
                         ACL_FLOAT, expected, shape, ACL_FLOAT, stream);
  }

  // Branch family: float*fp16 mixed path with end-to-end result validation.
  {
    const std::vector<float> selfData = {1.5f, -2.0f, 3.25f, -4.5f};
    const std::vector<uint16_t> otherData = {0x3c00, 0x4000, 0x4200, 0x4400};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> expected = ComputeFloat16FloatMulExpected(otherData, shape, selfData, shape);
    failed += RunMulCase("mul_float_fp16_promote_out_float", selfData, shape, ACL_FLOAT, otherData, shape,
                         ACL_FLOAT16, expected, shape, ACL_FLOAT, stream);
  }

  // Branch family: float*bf16 mixed path with end-to-end result validation.
  {
    const std::vector<float> selfData = {1.5f, -2.0f, 3.25f, -4.5f};
    const std::vector<uint16_t> otherData = {0x3f80, 0x4000, 0x4040, 0x4080};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> expected = ComputeFloatBFloat16MulExpected(selfData, shape, otherData, shape);
    failed += RunMulCase("mul_float_bf16_promote_out_float", selfData, shape, ACL_FLOAT, otherData, shape,
                         ACL_BF16, expected, shape, ACL_FLOAT, stream);
  }

  // Branch family: fp16*fp16 legal x16 path with exact raw-bit validation.
  {
    const std::vector<uint16_t> selfData = {0x3c00, 0x4000, 0x4400, 0x3800};
    const std::vector<uint16_t> otherData = {0x3c00, 0x4000, 0x3800, 0x4400};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint16_t> expected = ComputeFloat16MulExpected(selfData, shape, otherData, shape);
    failed += RunMulCase("mul_fp16_fp16_same_shape", selfData, shape, ACL_FLOAT16, otherData, shape,
                         ACL_FLOAT16, expected, shape, ACL_FLOAT16, stream);
  }

  // Branch family: bf16*bf16 legal x16 path with exact raw-bit validation.
  {
    const std::vector<uint16_t> selfData = {0x3f80, 0x4000, 0x4080, 0x3f00};
    const std::vector<uint16_t> otherData = {0x3f80, 0x4000, 0x3f00, 0x4080};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint16_t> expected = ComputeBFloat16MulExpected(selfData, shape, otherData, shape);
    failed += RunMulCase("mul_bf16_bf16_same_shape", selfData, shape, ACL_BF16, otherData, shape,
                         ACL_BF16, expected, shape, ACL_BF16, stream);
  }

  // Branch family: complex64 AiCore support path with end-to-end validation.
  {
    const std::vector<std::complex<float>> selfData = {{1.0f, 2.0f}, {-1.0f, 0.5f}};
    const std::vector<std::complex<float>> otherData = {{0.5f, -1.0f}, {2.0f, 1.0f}};
    const std::vector<int64_t> shape = {2};
    const std::vector<std::complex<float>> expected = ComputeMulExpectedDirect(selfData, shape, otherData, shape);
    failed += RunMulCase("mul_complex64_same_shape", selfData, shape, ACL_COMPLEX64, otherData,
                         shape, ACL_COMPLEX64, expected, shape, ACL_COMPLEX64, stream);
  }

  // Branch family: complex32 tiling entry with end-to-end non-zero validation.
  {
    const std::vector<uint32_t> selfData = {PackComplex32(1.0f, 2.0f), PackComplex32(-1.0f, 0.5f)};
    const std::vector<uint32_t> otherData = {PackComplex32(3.0f, 4.0f), PackComplex32(2.0f, -1.0f)};
    const std::vector<int64_t> shape = {2};
    const std::vector<uint32_t> outData(2, 0U);
    failed += RunMulExpectedFailureCase("mul_complex32_nonzero", selfData, shape, ACL_COMPLEX32, otherData,
                                        shape, ACL_COMPLEX32, outData, shape, ACL_COMPLEX32);
  }

  // Branch family: complex128 tensor-tensor path.
  {
    const std::vector<std::complex<double>> selfData = {{1.0, 2.0}, {-1.0, 0.5}};
    const std::vector<std::complex<double>> otherData = {{3.0, 4.0}, {2.0, -1.0}};
    const std::vector<int64_t> shape = {2};
    const std::vector<std::complex<double>> outData(2, {0.0, 0.0});
    failed += RunMulExpectedFailureCase("mul_complex128_same_shape", selfData, shape, ACL_COMPLEX128, otherData,
                                        shape, ACL_COMPLEX128, outData, shape, ACL_COMPLEX128);
  }

  // Branch family: empty tensor early return in aclnnMulGetWorkspaceSize.
  {
    const std::vector<float> selfData;
    const std::vector<float> otherData;
    const std::vector<float> outData;
    const std::vector<int64_t> shape = {0, 3};
    failed += RunMulCase("mul_empty_tensor_workspace_zero", selfData, shape, ACL_FLOAT, otherData, shape,
                         ACL_FLOAT, outData, shape, ACL_FLOAT, stream);
    failed += RunMulWorkspaceOnlyCase("mul_empty_tensor_zero_workspace_assert", selfData, shape, ACL_FLOAT, otherData,
                                      shape, ACL_FLOAT, outData, shape, ACL_FLOAT, true);
  }

  // Branch family: tensor-tensor compute with cast back to int32 output.
  {
    const std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> otherData = {2.0f, 3.0f, 4.0f, 5.0f};
    const std::vector<int32_t> outData(4, 0);
    const std::vector<int64_t> shape = {2, 2};
    failed += RunMulExpectedFailureCase("mul_float_out_int32", selfData, shape, ACL_FLOAT, otherData, shape,
                                        ACL_FLOAT, outData, shape, ACL_INT32);
  }

  // Branch family: non-contiguous tensor input path.
  failed += RunNonContiguousMulCase("mul_noncontiguous_input", stream);

  // Branch family: tensor*scalar API path.
  {
    const std::vector<float> selfData = {1.0f, -2.0f, 0.5f, 4.0f};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunMulsCase("muls_float32_scalar", selfData, shape, ACL_FLOAT, 2.5f,
                          ACL_FLOAT, ComputeMulsExpected<float, float>(selfData, 2.5f),
                          ACL_FLOAT, stream);
  }

  {
    const std::vector<float> selfData = {1.0f, -2.0f, 0.5f, 4.0f};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunMulsCase("muls_float_double_scalar", selfData, shape, ACL_FLOAT, 2.0,
                          ACL_DOUBLE, ComputeMulsExpected<float, float>(selfData, 2.0), ACL_FLOAT, stream);
  }

  {
    const std::vector<uint8_t> selfData = {1, 0, 1, 1};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> outData(4, 0.0f);
    failed += RunMulsExpectedFailureCase("muls_bool_double_scalar", selfData, shape, ACL_BOOL, 2.0,
                                         ACL_DOUBLE, outData, ACL_FLOAT);
  }

  {
    const std::vector<float> selfData = {1.0f, -2.0f, 0.5f, 4.0f};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<std::complex<float>> outData(4, {0.0f, 0.0f});
    failed += RunMulsExpectedFailureCase("muls_float_complex64_scalar", selfData, shape, ACL_FLOAT,
                                         std::complex<float>(2.0f, 0.0f), ACL_COMPLEX64, outData, ACL_COMPLEX64);
  }

  {
    const std::vector<std::complex<double>> selfData = {{1.0, 2.0}, {-1.0, 0.5}};
    const std::vector<int64_t> shape = {2};
    const std::vector<std::complex<double>> outData(2, {0.0, 0.0});
    failed += RunMulsExpectedFailureCase("muls_complex128_scalar", selfData, shape, ACL_COMPLEX128,
                                         std::complex<double>(2.0, 0.0), ACL_COMPLEX128, outData, ACL_COMPLEX128);
  }

  {
    const std::vector<int64_t> selfData = {1, -2, 3, -4};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunMulsCase("muls_int64_int32_scalar", selfData, shape, ACL_INT64, static_cast<int32_t>(3),
                          ACL_INT32, ComputeMulsExpected<int64_t, int64_t>(selfData, static_cast<int32_t>(3)),
                          ACL_INT64, stream);
  }

  failed += RunNonContiguousMulsCase("muls_noncontiguous_input", stream);

  // Branch family: scalar-shape tensor broadcast through aclnnMul itself.
  {
    const std::vector<float> selfData = {2.0f};
    const std::vector<float> otherData = {1.0f, -2.0f, 3.0f, -4.0f};
    const std::vector<float> expected = {2.0f, -4.0f, 6.0f, -8.0f};
    const std::vector<int64_t> selfShape = {};
    const std::vector<int64_t> otherShape = {2, 2};
    const std::vector<int64_t> outShape = {2, 2};
    failed += RunMulCase("mul_scalar_tensor_broadcast", selfData, selfShape, ACL_FLOAT, otherData, otherShape,
                         ACL_FLOAT, expected, outShape, ACL_FLOAT, stream);
  }

  // Branch family: fp16 tensor with float scalar using dedicated Muls path.
  {
    const std::vector<uint16_t> selfData = {0x3c00, 0x4000, 0x4400, 0x3800};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint16_t> outData(4, 0U);
    failed += RunMulsExpectedFailureCase("muls_fp16_float_scalar", selfData, shape, ACL_FLOAT16, 2.0f,
                                         ACL_FLOAT, outData, ACL_FLOAT16);
  }

  // Branch family: fp16 scalar promotion fallback to float when scalar is not exactly representable in fp16.
  {
    const std::vector<uint16_t> selfData = {0x3c00, 0x4000, 0x4400, 0x3800};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint16_t> outData(4, 0U);
    failed += RunMulsExpectedFailureCase("muls_fp16_inexact_float_scalar", selfData, shape,
                                         ACL_FLOAT16, 1.1f, ACL_FLOAT, outData, ACL_FLOAT16);
  }

  // Branch family: bf16 tensor with float scalar using dedicated Muls path.
  {
    const std::vector<uint16_t> selfData = {0x3f80, 0x4000, 0x4080, 0x3f00};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint16_t> outData(4, 0U);
    failed += RunMulsExpectedFailureCase("muls_bf16_float_scalar", selfData, shape, ACL_BF16, 2.0f,
                                         ACL_FLOAT, outData, ACL_BF16);
  }

  {
    const std::vector<uint16_t> selfData = {0x3f80, 0x4000, 0x4080, 0x3f00};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint16_t> outData(4, 0U);
    failed += RunMulsExpectedFailureCase("muls_bf16_double_scalar", selfData, shape, ACL_BF16, 2.0,
                                         ACL_DOUBLE, outData, ACL_BF16);
  }

  {
    const std::vector<uint16_t> selfData = {0x3c00, 0x4000, 0x4400, 0x3800};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint16_t> outData(4, 0U);
    failed += RunMulsExpectedFailureCase("muls_fp16_double_scalar", selfData, shape, ACL_FLOAT16, 2.0,
                                         ACL_DOUBLE, outData, ACL_FLOAT16);
  }

  // Branch family: bf16 scalar promotion fallback to float when scalar is not exactly representable in bf16.
  {
    const std::vector<uint16_t> selfData = {0x3f80, 0x4000, 0x4080, 0x3f00};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<uint16_t> outData(4, 0U);
    failed += RunMulsExpectedFailureCase("muls_bf16_inexact_float_scalar", selfData, shape,
                                         ACL_BF16, 1.1f, ACL_FLOAT, outData, ACL_BF16);
  }

  // Branch family: empty tensor early return in aclnnMulsGetWorkspaceSize.
  {
    const std::vector<float> selfData;
    const std::vector<float> outData;
    const std::vector<int64_t> shape = {0, 4};
    failed += RunMulsCase("muls_empty_tensor_workspace_zero", selfData, shape, ACL_FLOAT, 2.0f,
                          ACL_FLOAT, outData, ACL_FLOAT, stream);
    failed += RunMulsWorkspaceOnlyCase("muls_empty_tensor_zero_workspace_assert", selfData, shape, ACL_FLOAT, 2.0f,
                                       ACL_FLOAT, outData, ACL_FLOAT, true);
  }

  // Branch family: Muls result dtype mismatch.
  failed += RunMulsInvalidOutDtypeCase("muls_invalid_out_dtype");

  // Branch family: inplace tensor*tensor with broadcast.
  {
    const std::vector<float> selfData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> otherData = {2.0f, 4.0f, 8.0f};
    const std::vector<int64_t> selfShape = {2, 3};
    const std::vector<int64_t> otherShape = {3};
    failed += RunInplaceMulCase("inplace_mul_float32_broadcast", selfData, selfShape, ACL_FLOAT,
                                otherData, otherShape, ACL_FLOAT,
                                ComputeMulExpected<float, float, float>(selfData, selfShape, otherData, otherShape),
                                stream);
  }

  {
    const std::vector<float> selfData = {1.5f, -2.0f, 3.25f, -4.5f};
    const std::vector<uint16_t> otherData = {0x3c00, 0x4000, 0x4200, 0x4400};
    const std::vector<int64_t> shape = {2, 2};
    const std::vector<float> expected = ComputeFloat16FloatMulExpected(otherData, shape, selfData, shape);
    failed += RunInplaceMulCase("inplace_mul_float_fp16_promote", selfData, shape, ACL_FLOAT,
                                otherData, shape, ACL_FLOAT16, expected, stream);
  }

  // Branch family: empty tensor early return in aclnnInplaceMulGetWorkspaceSize.
  {
    const std::vector<float> selfData;
    const std::vector<float> otherData;
    const std::vector<int64_t> shape = {0, 2};
    failed += RunInplaceMulCase("inplace_mul_empty_tensor_workspace_zero", selfData, shape, ACL_FLOAT,
                                otherData, shape, ACL_FLOAT, std::vector<float>{}, stream);
    failed += RunInplaceMulWorkspaceOnlyCase("inplace_mul_empty_tensor_zero_workspace_assert", selfData, shape,
                                             ACL_FLOAT, otherData, shape, ACL_FLOAT, true);
  }

  // Branch family: InplaceMul mixed-dtype compute cast back into int32 selfRef.
  {
    const std::vector<int32_t> selfData = {1, 2, 3, 4};
    const std::vector<float> otherData = {2.0f, 2.0f, 2.0f, 2.0f};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunInplaceMulExpectedFailureCase("inplace_mul_int32_float_castback", selfData, shape, ACL_INT32,
                                               otherData, shape, ACL_FLOAT);
  }

  // Branch family: inplace tensor*scalar path.
  {
    const std::vector<float> selfData = {-1.0f, 2.0f, -3.0f, 4.0f};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunInplaceMulsCase("inplace_muls_float32_scalar", selfData, shape, ACL_FLOAT, -3.0f,
                                 ACL_FLOAT, ComputeMulsExpected<float, float>(selfData, -3.0f), stream);
  }

  {
    const std::vector<float> selfData = {1.0f, -2.0f, 0.5f, 4.0f};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunInplaceMulsCase("inplace_muls_float_double_scalar", selfData, shape, ACL_FLOAT, 2.0,
                                 ACL_DOUBLE, ComputeMulsExpected<float, float>(selfData, 2.0), stream);
  }

  {
    const std::vector<std::complex<float>> selfData = {{1.0f, 2.0f}, {-1.0f, 0.5f}};
    const std::vector<int64_t> shape = {2};
    const std::vector<std::complex<float>> expected = {{2.0f, 4.0f}, {-2.0f, 1.0f}};
    failed += RunInplaceMulsCase("inplace_muls_complex64_scalar", selfData, shape, ACL_COMPLEX64,
                                 std::complex<float>(2.0f, 0.0f), ACL_COMPLEX64, expected, stream);
  }

  {
    const std::vector<std::complex<double>> selfData = {{1.0, 2.0}, {-1.0, 0.5}};
    const std::vector<int64_t> shape = {2};
    failed += RunInplaceMulsExpectedFailureCase("inplace_muls_complex128_scalar", selfData, shape, ACL_COMPLEX128,
                                                std::complex<double>(2.0, 0.0), ACL_COMPLEX128);
  }

  {
    const std::vector<int64_t> selfData = {1, -2, 3, -4};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunInplaceMulsCase("inplace_muls_int64_int32_scalar", selfData, shape, ACL_INT64,
                                 static_cast<int32_t>(3), ACL_INT32,
                                 ComputeMulsExpected<int64_t, int64_t>(selfData, static_cast<int32_t>(3)), stream);
  }

  // Branch family: fp16 inplace tensor*float scalar path.
  {
    const std::vector<uint16_t> selfData = {0x3c00, 0x4000, 0x4400, 0x3800};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunInplaceMulsExpectedFailureCase("inplace_muls_fp16_float_scalar", selfData, shape,
                                                ACL_FLOAT16, 2.0f, ACL_FLOAT);
  }

  // Branch family: fp16 inplace scalar promotion fallback to float.
  {
    const std::vector<uint16_t> selfData = {0x3c00, 0x4000, 0x4400, 0x3800};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunInplaceMulsExpectedFailureCase("inplace_muls_fp16_inexact_float_scalar", selfData, shape,
                                                ACL_FLOAT16, 1.1f, ACL_FLOAT);
  }

  // Branch family: bf16 inplace tensor*float scalar path.
  {
    const std::vector<uint16_t> selfData = {0x3f80, 0x4000, 0x4080, 0x3f00};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunInplaceMulsExpectedFailureCase("inplace_muls_bf16_float_scalar", selfData, shape,
                                                ACL_BF16, 2.0f, ACL_FLOAT);
  }

  {
    const std::vector<uint16_t> selfData = {0x3f80, 0x4000, 0x4080, 0x3f00};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunInplaceMulsExpectedFailureCase("inplace_muls_bf16_double_scalar", selfData, shape,
                                                ACL_BF16, 2.0, ACL_DOUBLE);
  }

  {
    const std::vector<uint16_t> selfData = {0x3c00, 0x4000, 0x4400, 0x3800};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunInplaceMulsExpectedFailureCase("inplace_muls_fp16_double_scalar", selfData, shape,
                                                ACL_FLOAT16, 2.0, ACL_DOUBLE);
  }

  // Branch family: bf16 inplace scalar promotion fallback to float.
  {
    const std::vector<uint16_t> selfData = {0x3f80, 0x4000, 0x4080, 0x3f00};
    const std::vector<int64_t> shape = {2, 2};
    failed += RunInplaceMulsExpectedFailureCase("inplace_muls_bf16_inexact_float_scalar", selfData, shape,
                                                ACL_BF16, 1.1f, ACL_FLOAT);
  }

  // Branch family: empty tensor early return in aclnnInplaceMulsGetWorkspaceSize.
  {
    const std::vector<float> selfData;
    const std::vector<int64_t> shape = {0, 4};
    failed += RunInplaceMulsCase("inplace_muls_empty_tensor_workspace_zero", selfData, shape, ACL_FLOAT,
                                 2.0f, ACL_FLOAT, std::vector<float>{}, stream);
    failed += RunInplaceMulsWorkspaceOnlyCase("inplace_muls_empty_tensor_zero_workspace_assert", selfData, shape,
                                              ACL_FLOAT, 2.0f, ACL_FLOAT, true);
  }

  // Branch family: InplaceMuls promoted result cannot cast back to selfRef dtype.
  failed += RunInplaceMulsInvalidResultTypeCase("inplace_muls_invalid_result_dtype");

  // Phase-1 validation branches: null pointer, invalid broadcast, unsupported dtype, invalid inplace broadcast.
  failed += RunNullSelfCase("mul_null_self");
  failed += RunNullOtherCase("mul_null_other");
  failed += RunNullOutCase("mul_null_out");
  failed += RunInvalidBroadcastCase("mul_broadcast_invalid");
  failed += RunMulWrongOutShapeCase("mul_wrong_out_shape");
  failed += RunMulOverMaxDimCase("mul_over_max_dim");
  failed += RunUnsupportedDtypeCase("mul_unsupported_uint32_dtype");
  failed += RunInplaceMulNullSelfCase("inplace_mul_null_self");
  failed += RunInplaceMulNullOtherCase("inplace_mul_null_other");
  failed += RunInplaceInvalidBroadcastCase("inplace_mul_broadcast_expands_self_invalid");
  failed += RunMulsNullSelfCase("muls_null_self");
  failed += RunMulsNullScalarCase("muls_null_scalar");
  failed += RunMulsNullOutCase("muls_null_out");
  failed += RunMulsWrongOutShapeCase("muls_wrong_out_shape");
  failed += RunMulsUnsupportedDtypeCase("muls_unsupported_uint32_dtype");
  failed += RunInplaceMulsNullSelfCase("inplace_muls_null_self");
  failed += RunInplaceMulsNullScalarCase("inplace_muls_null_scalar");
  failed += RunInplaceMulsUnsupportedDtypeCase("inplace_muls_unsupported_uint32_dtype");

  LOG_PRINT("\n=== Summary: %d failed ===\n", failed);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return failed;
}
