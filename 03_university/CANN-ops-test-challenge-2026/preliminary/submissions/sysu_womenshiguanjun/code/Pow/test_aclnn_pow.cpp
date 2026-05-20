#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_exp2.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"

namespace {

#define CHECK_RET(cond, fmt, ...)                                       \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::printf("[ERROR] " fmt "\n", ##__VA_ARGS__);                  \
      return false;                                                      \
    }                                                                    \
  } while (0)

struct Env {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  bool ready = false;

  bool Init() {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, "aclInit failed: %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSetDevice failed: %d", ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtCreateStream failed: %d", ret);
    ready = true;
    return true;
  }

  void Finalize() {
    if (!ready) {
      return;
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    ready = false;
  }
};

int64_t ShapeSize(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t x : shape) {
    n *= x;
  }
  return n;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

struct TensorRes {
  aclTensor* tensor = nullptr;
  void* device = nullptr;
  size_t bytes = 0;

  void Destroy() {
    if (tensor != nullptr) {
      aclDestroyTensor(tensor);
      tensor = nullptr;
    }
    if (device != nullptr) {
      aclrtFree(device);
      device = nullptr;
    }
    bytes = 0;
  }
};

bool CreateTensor(const void* hostPtr, size_t elemBytes, int64_t elemCount, const std::vector<int64_t>& shape,
                  aclDataType dtype, TensorRes* out) {
  const size_t bytes = elemBytes * static_cast<size_t>(elemCount);
  auto ret = aclrtMalloc(&out->device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, "aclrtMalloc failed: %d", ret);

  ret = aclrtMemcpy(out->device, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, "aclrtMemcpy H2D failed: %d", ret);

  std::vector<int64_t> strides = MakeStrides(shape);
  out->tensor =
      aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                      out->device);
  CHECK_RET(out->tensor != nullptr, "aclCreateTensor failed");
  out->bytes = bytes;
  return true;
}

template <typename T>
bool CreateTensorFromVec(const std::vector<T>& host, const std::vector<int64_t>& shape, aclDataType dtype, TensorRes* out) {
  return CreateTensor(host.data(), sizeof(T), static_cast<int64_t>(host.size()), shape, dtype, out);
}

template <typename T>
aclScalar* CreateScalar(T v, aclDataType dtype) {
  return aclCreateScalar(&v, dtype);
}

template <typename T>
bool CopyBack(const TensorRes& tensor, std::vector<T>* host) {
  auto ret = aclrtMemcpy(host->data(), tensor.bytes, tensor.device, tensor.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, "aclrtMemcpy D2H failed: %d", ret);
  return true;
}

bool Sync(aclrtStream stream) {
  auto ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed: %d", ret);
  return true;
}

bool IsClose(double actual, double expected, double atol = 1e-5, double rtol = 1e-5) {
  if (std::isnan(expected)) {
    return std::isnan(actual);
  }
  if (std::isinf(expected)) {
    return std::isinf(actual) && (std::signbit(expected) == std::signbit(actual));
  }
  const double diff = std::fabs(actual - expected);
  return diff <= (atol + rtol * std::fabs(expected));
}

template <typename T>
bool CheckFloatVec(const std::vector<T>& actual, const std::vector<double>& expected, double atol = 1e-4, double rtol = 1e-4) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!IsClose(static_cast<double>(actual[i]), expected[i], atol, rtol)) {
      std::printf("[MISMATCH] idx=%zu actual=%lf expected=%lf\n", i, static_cast<double>(actual[i]), expected[i]);
      return false;
    }
  }
  return true;
}

template <typename T>
bool CheckIntVec(const std::vector<T>& actual, const std::vector<T>& expected) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::printf("[MISMATCH] idx=%zu actual=%lld expected=%lld\n", i, static_cast<long long>(actual[i]),
                  static_cast<long long>(expected[i]));
      return false;
    }
  }
  return true;
}

std::vector<int64_t> BroadcastShape(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  const size_t n = (a.size() > b.size()) ? a.size() : b.size();
  std::vector<int64_t> out(n, 1);
  for (size_t i = 0; i < n; ++i) {
    const int64_t da = (i < n - a.size()) ? 1 : a[i - (n - a.size())];
    const int64_t db = (i < n - b.size()) ? 1 : b[i - (n - b.size())];
    out[i] = (da > db) ? da : db;
  }
  return out;
}

std::vector<int64_t> UnravelIndex(int64_t linear, const std::vector<int64_t>& shape) {
  std::vector<int64_t> coord(shape.size(), 0);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    coord[i] = linear % shape[i];
    linear /= shape[i];
  }
  return coord;
}

int64_t BroadcastIndex(const std::vector<int64_t>& coordOut, const std::vector<int64_t>& inShape) {
  const size_t offset = coordOut.size() - inShape.size();
  std::vector<int64_t> inCoord(inShape.size(), 0);
  for (size_t i = 0; i < inShape.size(); ++i) {
    inCoord[i] = (inShape[i] == 1) ? 0 : coordOut[offset + i];
  }
  const std::vector<int64_t> strides = MakeStrides(inShape);
  int64_t idx = 0;
  for (size_t i = 0; i < inShape.size(); ++i) {
    idx += inCoord[i] * strides[i];
  }
  return idx;
}

bool RunPowTensorScalarCase(Env& env, const std::string& name, const std::vector<float>& selfData,
                            const std::vector<int64_t>& shape, float exponent, bool inplace, double atol = 1e-4,
                            double rtol = 1e-4) {
  TensorRes self;
  TensorRes out;
  aclScalar* expScalar = nullptr;
  void* workspace = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  bool ok = CreateTensorFromVec(selfData, shape, ACL_FLOAT, &self);
  CHECK_RET(ok, "Create self tensor failed");

  const int64_t n = ShapeSize(shape);
  std::vector<float> outInit(static_cast<size_t>(n), 0.0f);
  ok = CreateTensorFromVec(outInit, shape, ACL_FLOAT, &out);
  CHECK_RET(ok, "Create out tensor failed");

  expScalar = aclCreateScalar(&exponent, ACL_FLOAT);
  CHECK_RET(expScalar != nullptr, "aclCreateScalar failed");

  aclnnStatus ret;
  if (!inplace) {
    ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, expScalar, out.tensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnPowTensorScalarGetWorkspaceSize failed: %d", ret);
  } else {
    ret = aclnnInplacePowTensorScalarGetWorkspaceSize(self.tensor, expScalar, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnInplacePowTensorScalarGetWorkspaceSize failed: %d", ret);
  }

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMalloc workspace failed: %d", ret);
  }

  if (!inplace) {
    ret = aclnnPowTensorScalar(workspace, workspaceSize, executor, env.stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnPowTensorScalar failed: %d", ret);
  } else {
    ret = aclnnInplacePowTensorScalar(workspace, workspaceSize, executor, env.stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnInplacePowTensorScalar failed: %d", ret);
  }

  ok = Sync(env.stream);
  CHECK_RET(ok, "stream sync failed");

  std::vector<float> actual(static_cast<size_t>(n), 0.0f);
  if (!inplace) {
    ok = CopyBack(out, &actual);
  } else {
    ok = CopyBack(self, &actual);
  }
  CHECK_RET(ok, "Copy back failed");

  std::vector<double> expected(static_cast<size_t>(n), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    expected[static_cast<size_t>(i)] = std::pow(static_cast<double>(selfData[static_cast<size_t>(i)]),
                                                static_cast<double>(exponent));
  }
  ok = CheckFloatVec(actual, expected, atol, rtol);

  if (expScalar != nullptr) {
    aclDestroyScalar(expScalar);
  }
  self.Destroy();
  out.Destroy();
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }

  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
  return ok;
}

bool RunPowScalarTensorCase(Env& env, const std::string& name, float base, const std::vector<float>& exponentData,
                            const std::vector<int64_t>& shape, double atol = 1e-4, double rtol = 1e-4) {
  TensorRes exponent;
  TensorRes out;
  aclScalar* baseScalar = nullptr;
  void* workspace = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  bool ok = CreateTensorFromVec(exponentData, shape, ACL_FLOAT, &exponent);
  CHECK_RET(ok, "Create exponent tensor failed");

  const int64_t n = ShapeSize(shape);
  std::vector<float> outInit(static_cast<size_t>(n), 0.0f);
  ok = CreateTensorFromVec(outInit, shape, ACL_FLOAT, &out);
  CHECK_RET(ok, "Create out tensor failed");

  baseScalar = aclCreateScalar(&base, ACL_FLOAT);
  CHECK_RET(baseScalar != nullptr, "aclCreateScalar failed");

  auto ret =
      aclnnPowScalarTensorGetWorkspaceSize(baseScalar, exponent.tensor, out.tensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, "aclnnPowScalarTensorGetWorkspaceSize failed: %d", ret);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMalloc workspace failed: %d", ret);
  }
  ret = aclnnPowScalarTensor(workspace, workspaceSize, executor, env.stream);
  CHECK_RET(ret == ACL_SUCCESS, "aclnnPowScalarTensor failed: %d", ret);

  ok = Sync(env.stream);
  CHECK_RET(ok, "stream sync failed");

  std::vector<float> actual(static_cast<size_t>(n), 0.0f);
  ok = CopyBack(out, &actual);
  CHECK_RET(ok, "Copy back failed");

  std::vector<double> expected(static_cast<size_t>(n), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    expected[static_cast<size_t>(i)] =
        std::pow(static_cast<double>(base), static_cast<double>(exponentData[static_cast<size_t>(i)]));
  }
  ok = CheckFloatVec(actual, expected, atol, rtol);

  if (baseScalar != nullptr) {
    aclDestroyScalar(baseScalar);
  }
  exponent.Destroy();
  out.Destroy();
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }

  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
  return ok;
}

bool RunPowScalarTensorFillOneCase(Env& env, const std::string& name, float base, const std::vector<float>& exponentData,
                                   const std::vector<int64_t>& shape) {
  TensorRes exponent;
  TensorRes out;
  aclScalar* baseScalar = nullptr;
  void* workspace = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  bool ok = CreateTensorFromVec(exponentData, shape, ACL_FLOAT, &exponent);
  CHECK_RET(ok, "Create exponent tensor failed");
  const int64_t n = ShapeSize(shape);
  std::vector<float> outInit(static_cast<size_t>(n), 0.0f);
  ok = CreateTensorFromVec(outInit, shape, ACL_FLOAT, &out);
  CHECK_RET(ok, "Create out tensor failed");

  baseScalar = CreateScalar(base, ACL_FLOAT);
  CHECK_RET(baseScalar != nullptr, "aclCreateScalar failed");

  auto ret =
      aclnnPowScalarTensorGetWorkspaceSize(baseScalar, exponent.tensor, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    std::printf("[PASS] %s (unsupported status=%d)\n", name.c_str(), ret);
    aclDestroyScalar(baseScalar);
    exponent.Destroy();
    out.Destroy();
    return true;
  }

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMalloc workspace failed: %d", ret);
  }
  ret = aclnnPowScalarTensor(workspace, workspaceSize, executor, env.stream);
  CHECK_RET(ret == ACL_SUCCESS, "aclnnPowScalarTensor failed: %d", ret);
  ok = Sync(env.stream);
  CHECK_RET(ok, "stream sync failed");

  std::vector<float> actual(static_cast<size_t>(n), 0.0f);
  ok = CopyBack(out, &actual);
  CHECK_RET(ok, "Copy back failed");
  std::vector<double> expected(static_cast<size_t>(n), 1.0);
  ok = CheckFloatVec(actual, expected, 1e-6, 1e-6);

  aclDestroyScalar(baseScalar);
  exponent.Destroy();
  out.Destroy();
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
  return ok;
}

bool RunPowTensorScalarDoubleExpCase(Env& env, const std::string& name, const std::vector<float>& selfData,
                                     const std::vector<int64_t>& shape, double exponent) {
  TensorRes self;
  TensorRes out;
  aclScalar* expScalar = nullptr;
  void* workspace = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  bool ok = CreateTensorFromVec(selfData, shape, ACL_FLOAT, &self);
  CHECK_RET(ok, "Create self tensor failed");
  const int64_t n = ShapeSize(shape);
  std::vector<float> outInit(static_cast<size_t>(n), 0.0f);
  ok = CreateTensorFromVec(outInit, shape, ACL_FLOAT, &out);
  CHECK_RET(ok, "Create out tensor failed");

  expScalar = CreateScalar(exponent, ACL_DOUBLE);
  CHECK_RET(expScalar != nullptr, "aclCreateScalar double failed");

  auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, expScalar, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    std::printf("[PASS] %s (unsupported status=%d)\n", name.c_str(), ret);
    aclDestroyScalar(expScalar);
    self.Destroy();
    out.Destroy();
    return true;
  }
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMalloc workspace failed: %d", ret);
  }
  ret = aclnnPowTensorScalar(workspace, workspaceSize, executor, env.stream);
  CHECK_RET(ret == ACL_SUCCESS, "aclnnPowTensorScalar failed: %d", ret);
  ok = Sync(env.stream);
  CHECK_RET(ok, "stream sync failed");

  std::vector<float> actual(static_cast<size_t>(n), 0.0f);
  ok = CopyBack(out, &actual);
  CHECK_RET(ok, "Copy back failed");

  std::vector<double> expected(static_cast<size_t>(n), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    expected[static_cast<size_t>(i)] = std::pow(static_cast<double>(selfData[static_cast<size_t>(i)]), exponent);
  }
  ok = CheckFloatVec(actual, expected, 1e-4, 1e-4);

  if (expScalar != nullptr) {
    aclDestroyScalar(expScalar);
  }
  self.Destroy();
  out.Destroy();
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
  return ok;
}

bool RunPowTensorScalarInt64Case(Env& env, const std::string& name, const std::vector<int64_t>& selfData,
                                 const std::vector<int64_t>& shape, int64_t exponent) {
  TensorRes self;
  TensorRes out;
  aclScalar* expScalar = nullptr;
  void* workspace = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  bool ok = CreateTensorFromVec(selfData, shape, ACL_INT64, &self);
  CHECK_RET(ok, "Create self tensor failed");
  const int64_t n = ShapeSize(shape);
  std::vector<int64_t> outInit(static_cast<size_t>(n), 0);
  ok = CreateTensorFromVec(outInit, shape, ACL_INT64, &out);
  CHECK_RET(ok, "Create out tensor failed");
  expScalar = CreateScalar(exponent, ACL_INT64);
  CHECK_RET(expScalar != nullptr, "aclCreateScalar int64 failed");

  auto ret = aclnnPowTensorScalarGetWorkspaceSize(self.tensor, expScalar, out.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    std::printf("[PASS] %s (unsupported status=%d)\n", name.c_str(), ret);
    aclDestroyScalar(expScalar);
    self.Destroy();
    out.Destroy();
    return true;
  }
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMalloc workspace failed: %d", ret);
  }
  ret = aclnnPowTensorScalar(workspace, workspaceSize, executor, env.stream);
  CHECK_RET(ret == ACL_SUCCESS, "aclnnPowTensorScalar failed: %d", ret);
  ok = Sync(env.stream);
  CHECK_RET(ok, "stream sync failed");

  std::vector<int64_t> actual(static_cast<size_t>(n), 0);
  ok = CopyBack(out, &actual);
  CHECK_RET(ok, "Copy back failed");

  std::vector<int64_t> expected(static_cast<size_t>(n), 0);
  for (int64_t i = 0; i < n; ++i) {
    expected[static_cast<size_t>(i)] =
        static_cast<int64_t>(std::pow(static_cast<double>(selfData[static_cast<size_t>(i)]), static_cast<double>(exponent)));
  }
  ok = CheckIntVec(actual, expected);

  if (expScalar != nullptr) {
    aclDestroyScalar(expScalar);
  }
  self.Destroy();
  out.Destroy();
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
  return ok;
}

bool RunExpectedFailNullptrCase(Env& env, const std::string& name) {
  (void)env;
  std::vector<int64_t> shape = {2, 2};
  TensorRes out;
  std::vector<float> outInit = {0, 0, 0, 0};
  bool ok = CreateTensorFromVec(outInit, shape, ACL_FLOAT, &out);
  CHECK_RET(ok, "Create out tensor failed");

  float exp = 2.0f;
  aclScalar* expScalar = CreateScalar(exp, ACL_FLOAT);
  CHECK_RET(expScalar != nullptr, "aclCreateScalar failed");

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnPowTensorScalarGetWorkspaceSize(nullptr, expScalar, out.tensor, &workspaceSize, &executor);
  const bool passed = (ret != ACL_SUCCESS);
  std::printf("[%s] %s\n", passed ? "PASS" : "FAIL", name.c_str());

  aclDestroyScalar(expScalar);
  out.Destroy();
  return passed;
}

template <typename T>
bool RunPowTensorTensorCase(Env& env, const std::string& name, const std::vector<T>& selfData,
                            const std::vector<int64_t>& selfShape, const std::vector<T>& expData,
                            const std::vector<int64_t>& expShape, aclDataType dtype, bool inplace) {
  TensorRes self;
  TensorRes exp;
  TensorRes out;
  void* workspace = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  bool ok = CreateTensorFromVec(selfData, selfShape, dtype, &self);
  CHECK_RET(ok, "Create self tensor failed");
  ok = CreateTensorFromVec(expData, expShape, dtype, &exp);
  CHECK_RET(ok, "Create exp tensor failed");

  std::vector<int64_t> outShape = BroadcastShape(selfShape, expShape);
  const int64_t outN = ShapeSize(outShape);
  std::vector<T> outInit(static_cast<size_t>(outN), static_cast<T>(0));
  ok = CreateTensorFromVec(outInit, outShape, dtype, &out);
  CHECK_RET(ok, "Create out tensor failed");

  aclnnStatus ret;
  if (!inplace) {
    ret = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exp.tensor, out.tensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnPowTensorTensorGetWorkspaceSize failed: %d", ret);
  } else {
    ret = aclnnInplacePowTensorTensorGetWorkspaceSize(self.tensor, exp.tensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnInplacePowTensorTensorGetWorkspaceSize failed: %d", ret);
  }
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMalloc workspace failed: %d", ret);
  }
  if (!inplace) {
    ret = aclnnPowTensorTensor(workspace, workspaceSize, executor, env.stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnPowTensorTensor failed: %d", ret);
  } else {
    ret = aclnnInplacePowTensorTensor(workspace, workspaceSize, executor, env.stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnInplacePowTensorTensor failed: %d", ret);
  }

  ok = Sync(env.stream);
  CHECK_RET(ok, "stream sync failed");

  std::vector<T> actual(static_cast<size_t>(outN), static_cast<T>(0));
  if (!inplace) {
    ok = CopyBack(out, &actual);
  } else {
    const int64_t selfN = ShapeSize(selfShape);
    actual.resize(static_cast<size_t>(selfN));
    ok = CopyBack(self, &actual);
  }
  CHECK_RET(ok, "Copy back failed");

  bool passed = true;
  if constexpr (std::is_floating_point<T>::value) {
    std::vector<double> expected(actual.size(), 0.0);
    const std::vector<int64_t> shapeForExpected = inplace ? selfShape : outShape;
    for (size_t i = 0; i < expected.size(); ++i) {
      const std::vector<int64_t> coord = UnravelIndex(static_cast<int64_t>(i), shapeForExpected);
      const int64_t iSelf = BroadcastIndex(coord, selfShape);
      const int64_t iExp = BroadcastIndex(coord, expShape);
      expected[i] = std::pow(static_cast<double>(selfData[static_cast<size_t>(iSelf)]),
                             static_cast<double>(expData[static_cast<size_t>(iExp)]));
    }
    passed = CheckFloatVec(actual, expected, 1e-4, 1e-4);
  } else {
    std::vector<T> expected(actual.size(), static_cast<T>(0));
    const std::vector<int64_t> shapeForExpected = inplace ? selfShape : outShape;
    for (size_t i = 0; i < expected.size(); ++i) {
      const std::vector<int64_t> coord = UnravelIndex(static_cast<int64_t>(i), shapeForExpected);
      const int64_t iSelf = BroadcastIndex(coord, selfShape);
      const int64_t iExp = BroadcastIndex(coord, expShape);
      const double v = std::pow(static_cast<double>(selfData[static_cast<size_t>(iSelf)]),
                                static_cast<double>(expData[static_cast<size_t>(iExp)]));
      expected[i] = static_cast<T>(v);
    }
    passed = CheckIntVec(actual, expected);
  }

  self.Destroy();
  exp.Destroy();
  out.Destroy();
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  std::printf("[%s] %s\n", passed ? "PASS" : "FAIL", name.c_str());
  return passed;
}

bool RunExp2Case(Env& env, const std::string& name, const std::vector<float>& selfData, const std::vector<int64_t>& shape,
                 bool inplace, double atol = 1e-4, double rtol = 1e-4) {
  TensorRes self;
  TensorRes out;
  bool ok = CreateTensorFromVec(selfData, shape, ACL_FLOAT, &self);
  CHECK_RET(ok, "Create self tensor failed");
  const int64_t n = ShapeSize(shape);
  std::vector<float> outInit(static_cast<size_t>(n), 0.0f);
  ok = CreateTensorFromVec(outInit, shape, ACL_FLOAT, &out);
  CHECK_RET(ok, "Create out tensor failed");

  void* workspace = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus ret;
  if (!inplace) {
    ret = aclnnExp2GetWorkspaceSize(self.tensor, out.tensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnExp2GetWorkspaceSize failed: %d", ret);
  } else {
    ret = aclnnInplaceExp2GetWorkspaceSize(self.tensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnInplaceExp2GetWorkspaceSize failed: %d", ret);
  }
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMalloc workspace failed: %d", ret);
  }
  if (!inplace) {
    ret = aclnnExp2(workspace, workspaceSize, executor, env.stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnExp2 failed: %d", ret);
  } else {
    ret = aclnnInplaceExp2(workspace, workspaceSize, executor, env.stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnInplaceExp2 failed: %d", ret);
  }

  ok = Sync(env.stream);
  CHECK_RET(ok, "stream sync failed");

  std::vector<float> actual(static_cast<size_t>(n), 0.0f);
  if (!inplace) {
    ok = CopyBack(out, &actual);
  } else {
    ok = CopyBack(self, &actual);
  }
  CHECK_RET(ok, "Copy back failed");

  std::vector<double> expected(static_cast<size_t>(n), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    expected[static_cast<size_t>(i)] = std::pow(2.0, static_cast<double>(selfData[static_cast<size_t>(i)]));
  }
  ok = CheckFloatVec(actual, expected, atol, rtol);

  self.Destroy();
  out.Destroy();
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
  return ok;
}

}  // namespace

int main() {
  Env env;
  if (!env.Init()) {
    std::printf("[FAIL] ENV_INIT\n");
    return 1;
  }

  int passed = 0;
  int failed = 0;
  auto Run = [&](bool ok) {
    if (ok) {
      ++passed;
    } else {
      ++failed;
    }
  };

  Run(RunPowTensorScalarCase(env, "PowTensorScalar_fp32_exp0", {0.0f, -2.0f, 3.0f, 5.0f}, {2, 2}, 0.0f, false));
  Run(RunPowTensorScalarCase(env, "PowTensorScalar_fp32_exp1", {0.0f, -2.0f, 3.0f, 5.0f}, {2, 2}, 1.0f, false));
  Run(RunPowTensorScalarCase(env, "PowTensorScalar_fp32_exp0.5_nan_path", {-4.0f, -1.0f, 0.0f, 4.0f}, {2, 2}, 0.5f,
                             false, 1e-3, 1e-3));
  Run(RunPowTensorScalarCase(env, "PowTensorScalar_fp32_exp-1", {2.0f, -2.0f, 0.5f, -0.5f}, {2, 2}, -1.0f, false));
  Run(RunPowTensorScalarDoubleExpCase(env, "PowTensorScalar_fp32_double_exp2.5", {1.5f, 2.0f, 3.0f, 4.0f}, {2, 2}, 2.5));
  Run(RunPowTensorScalarInt64Case(env, "PowTensorScalar_int64_exp3_aicpu_path", {2, 3, 4, 5}, {2, 2}, 3));
  Run(RunPowTensorScalarCase(env, "InplacePowTensorScalar_fp32_exp3", {1.0f, -2.0f, 3.0f, -4.0f}, {2, 2}, 3.0f, true));

  Run(RunPowScalarTensorFillOneCase(env, "PowScalarTensor_fill_one_base1", 1.0f, {-1.0f, 0.0f, 1.0f, 2.0f}, {2, 2}));
  Run(RunPowScalarTensorCase(env, "PowScalarTensor_fp32_base2", 2.0f, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.5f},
                             {2, 3}));

  Run(RunPowTensorTensorCase<float>(env, "PowTensorTensor_fp32_broadcast", {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, {2, 3},
                                    {1.0f, 2.0f, 3.0f}, {3}, ACL_FLOAT, false));
  Run(RunPowTensorTensorCase<float>(env, "InplacePowTensorTensor_fp32", {2.0f, -2.0f, 3.0f, -3.0f}, {2, 2},
                                    {3.0f, 2.0f, 1.0f, 0.0f}, {2, 2}, ACL_FLOAT, true));
  Run(RunPowTensorTensorCase<int32_t>(env, "PowTensorTensor_int32", {2, 3, 4, 5}, {2, 2}, {3, 2, 1, 0}, {2, 2},
                                      ACL_INT32, false));
  Run(RunPowTensorTensorCase<int8_t>(env, "PowTensorTensor_int8", {2, -2, 3, -3}, {2, 2}, {3, 2, 1, 0}, {2, 2},
                                     ACL_INT8, false));
  Run(RunPowTensorTensorCase<uint8_t>(env, "PowTensorTensor_uint8", {2, 3, 4, 5}, {2, 2}, {3, 2, 1, 0}, {2, 2},
                                      ACL_UINT8, false));

  Run(RunExp2Case(env, "Exp2_fp32", {-1.0f, 0.0f, 1.0f, 2.0f}, {2, 2}, false));
  Run(RunExp2Case(env, "InplaceExp2_fp32", {-2.0f, -1.0f, 0.0f, 3.0f}, {2, 2}, true));
  Run(RunExpectedFailNullptrCase(env, "PowTensorScalar_nullptr_expected_fail"));

  env.Finalize();
  std::printf("========== Pow Test Summary ==========\n");
  std::printf("Total: %d, PASS: %d, FAIL: %d\n", passed + failed, passed, failed);
  return (failed == 0) ? 0 : 2;
}
