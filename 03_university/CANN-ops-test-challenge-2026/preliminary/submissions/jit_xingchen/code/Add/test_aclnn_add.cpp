/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

#define LOG_PRINT(fmt, ...)     \
  do {                          \
    std::printf(fmt, ##__VA_ARGS__); \
  } while (0)

static int g_pass = 0;
static int g_fail = 0;

struct CompareStats {
  bool pass = true;
  double maxAbs = 0.0;
  double maxRel = 0.0;
  int64_t badIndex = -1;
};

struct TensorHolder {
  void *addr = nullptr;
  aclTensor *tensor = nullptr;
  TensorHolder() = default;
  TensorHolder(const TensorHolder &) = delete;
  TensorHolder &operator=(const TensorHolder &) = delete;
  ~TensorHolder()
  {
    if (tensor != nullptr) {
      aclDestroyTensor(tensor);
      tensor = nullptr;
    }
    if (addr != nullptr) {
      aclrtFree(addr);
      addr = nullptr;
    }
  }
};

struct ScalarHolder {
  aclScalar *v = nullptr;
  ScalarHolder() = default;
  ScalarHolder(const ScalarHolder &) = delete;
  ScalarHolder &operator=(const ScalarHolder &) = delete;
  ~ScalarHolder()
  {
    if (v != nullptr) {
      aclDestroyScalar(v);
      v = nullptr;
    }
  }
};

struct WorkspaceHolder {
  void *addr = nullptr;
  WorkspaceHolder() = default;
  WorkspaceHolder(const WorkspaceHolder &) = delete;
  WorkspaceHolder &operator=(const WorkspaceHolder &) = delete;
  ~WorkspaceHolder()
  {
    if (addr != nullptr) {
      aclrtFree(addr);
      addr = nullptr;
    }
  }
};

static const char *DTypeStr(aclDataType dt)
{
  switch (dt) {
    case ACL_FLOAT:
      return "FLOAT";
    case ACL_FLOAT16:
      return "FLOAT16";
    case ACL_BF16:
      return "BF16";
    case ACL_DOUBLE:
      return "DOUBLE";
    case ACL_INT8:
      return "INT8";
    case ACL_UINT8:
      return "UINT8";
    case ACL_INT16:
      return "INT16";
    case ACL_INT32:
      return "INT32";
    case ACL_INT64:
      return "INT64";
    case ACL_BOOL:
      return "BOOL";
    case ACL_COMPLEX32:
      return "COMPLEX32";
    case ACL_COMPLEX64:
      return "COMPLEX64";
    case ACL_COMPLEX128:
      return "COMPLEX128";
    default:
      return "UNKNOWN";
  }
}

static std::string ShapeStr(const std::vector<int64_t> &shape)
{
  if (shape.empty()) {
    return "[]";
  }
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    s += std::to_string(shape[i]);
    if (i + 1 != shape.size()) {
      s += ", ";
    }
  }
  s += "]";
  return s;
}

static int64_t Numel(const std::vector<int64_t> &shape)
{
  if (shape.empty()) {
    return 1;
  }
  int64_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) {
      return -1;
    }
    n *= shape[i];
  }
  return n;
}

static std::vector<int64_t> MakeStrides(const std::vector<int64_t> &shape)
{
  std::vector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

static bool BroadcastShape(const std::vector<int64_t> &a,
                           const std::vector<int64_t> &b,
                           std::vector<int64_t> *out)
{
  if (out == nullptr) {
    return false;
  }
  out->clear();
  const size_t ra = a.size();
  const size_t rb = b.size();
  const size_t ro = (ra > rb) ? ra : rb;
  out->resize(ro, 1);
  for (size_t i = 0; i < ro; ++i) {
    const int64_t da = (i < ro - ra) ? 1 : a[i - (ro - ra)];
    const int64_t db = (i < ro - rb) ? 1 : b[i - (ro - rb)];
    if (da != db && da != 1 && db != 1) {
      return false;
    }
    (*out)[i] = (da > db) ? da : db;
  }
  return true;
}

static void UnravelIndex(int64_t flat,
                         const std::vector<int64_t> &shape,
                         std::vector<int64_t> *idx)
{
  idx->assign(shape.size(), 0);
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    const int64_t d = shape[static_cast<size_t>(i)];
    if (d > 0) {
      (*idx)[static_cast<size_t>(i)] = flat % d;
      flat /= d;
    }
  }
}

static int64_t BroadcastOffset(const std::vector<int64_t> &outIdx,
                               const std::vector<int64_t> &inShape,
                               const std::vector<int64_t> &inStrides)
{
  if (inShape.empty()) {
    return 0;
  }
  int64_t off = 0;
  const size_t ro = outIdx.size();
  const size_t ri = inShape.size();
  for (size_t i = 0; i < ri; ++i) {
    const size_t outPos = ro - ri + i;
    const int64_t id = (inShape[i] == 1) ? 0 : outIdx[outPos];
    off += id * inStrides[i];
  }
  return off;
}

static float F16ToF32(uint16_t h)
{
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t out = 0;

  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      int shift = 0;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        ++shift;
      }
      mant &= 0x3FFu;
      out = sign | (static_cast<uint32_t>(127 - 15 - shift) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    out = sign | 0x7F800000u | (mant << 13);
  } else {
    out = sign | ((exp + 112u) << 23) | (mant << 13);
  }

  float f = 0.0f;
  std::memcpy(&f, &out, sizeof(float));
  return f;
}

static uint16_t F32ToF16(float f)
{
  uint32_t x = 0;
  std::memcpy(&x, &f, sizeof(float));
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;

  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    mant |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t hmant = mant >> shift;
    if ((mant >> (shift - 1)) & 1u) {
      hmant++;
    }
    return static_cast<uint16_t>(sign | hmant);
  }

  if (exp >= 31) {
    const bool isNan = ((x & 0x7FFFFFFFu) > 0x7F800000u);
    return static_cast<uint16_t>(isNan ? (sign | 0x7E00u) : (sign | 0x7C00u));
  }

  mant += 0x1000u;
  if (mant & 0x800000u) {
    mant = 0;
    exp++;
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }

  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

static float BF16ToF32(uint16_t b)
{
  uint32_t x = static_cast<uint32_t>(b) << 16;
  float f = 0.0f;
  std::memcpy(&f, &x, sizeof(float));
  return f;
}

static uint16_t F32ToBF16(float f)
{
  uint32_t x = 0;
  std::memcpy(&x, &f, sizeof(float));
  const uint32_t lsb = (x >> 16) & 1u;
  x += 0x7FFFu + lsb;
  return static_cast<uint16_t>(x >> 16);
}

template <typename T>
static double HostValToDouble(T v, aclDataType dt)
{
  if (dt == ACL_FLOAT16) {
    return static_cast<double>(F16ToF32(static_cast<uint16_t>(v)));
  }
  if (dt == ACL_BF16) {
    return static_cast<double>(BF16ToF32(static_cast<uint16_t>(v)));
  }
  if (dt == ACL_BOOL) {
    return (v != 0) ? 1.0 : 0.0;
  }
  return static_cast<double>(v);
}

static double QuantizeByDType(double x, aclDataType dt)
{
  switch (dt) {
    case ACL_FLOAT16: {
      return static_cast<double>(F16ToF32(F32ToF16(static_cast<float>(x))));
    }
    case ACL_BF16: {
      return static_cast<double>(BF16ToF32(F32ToBF16(static_cast<float>(x))));
    }
    case ACL_FLOAT:
      return static_cast<double>(static_cast<float>(x));
    case ACL_DOUBLE:
      return x;
    case ACL_INT8:
      return static_cast<double>(static_cast<int8_t>(x));
    case ACL_UINT8:
      return static_cast<double>(static_cast<uint8_t>(x));
    case ACL_INT16:
      return static_cast<double>(static_cast<int16_t>(x));
    case ACL_INT32:
      return static_cast<double>(static_cast<int32_t>(x));
    case ACL_INT64:
      return static_cast<double>(static_cast<int64_t>(x));
    case ACL_BOOL:
      return (x != 0.0) ? 1.0 : 0.0;
    default:
      return x;
  }
}

static void GetTolerance(aclDataType dt, double *atol, double *rtol)
{
  if (dt == ACL_FLOAT16 || dt == ACL_BF16) {
    *atol = 2e-3;
    *rtol = 2e-3;
  } else if (dt == ACL_FLOAT) {
    *atol = 1e-5;
    *rtol = 1e-4;
  } else if (dt == ACL_DOUBLE) {
    *atol = 1e-12;
    *rtol = 1e-12;
  } else {
    *atol = 0.0;
    *rtol = 0.0;
  }
}

static CompareStats CompareVec(const std::vector<double> &expected,
                               const std::vector<double> &actual,
                               double atol,
                               double rtol)
{
  CompareStats s;
  if (expected.size() != actual.size()) {
    s.pass = false;
    return s;
  }

  for (size_t i = 0; i < expected.size(); ++i) {
    const double e = expected[i];
    const double a = actual[i];

    const bool en = std::isnan(e);
    const bool an = std::isnan(a);
    if (en || an) {
      if (!(en && an)) {
        s.pass = false;
        s.badIndex = static_cast<int64_t>(i);
        return s;
      }
      continue;
    }

    const bool ei = std::isinf(e);
    const bool ai = std::isinf(a);
    if (ei || ai) {
      const bool ok = ei && ai && (std::signbit(e) == std::signbit(a));
      if (!ok) {
        s.pass = false;
        s.badIndex = static_cast<int64_t>(i);
        return s;
      }
      continue;
    }

    const double absErr = std::fabs(a - e);
    const double relErr = absErr / std::fmax(1.0, std::fabs(e));
    s.maxAbs = std::fmax(s.maxAbs, absErr);
    s.maxRel = std::fmax(s.maxRel, relErr);

    if (!(absErr <= (atol + rtol * std::fabs(e)))) {
      s.pass = false;
      s.badIndex = static_cast<int64_t>(i);
      return s;
    }
  }

  return s;
}

static void Record(const std::string &name,
                   bool ok,
                   double maxAbs = 0.0,
                   double maxRel = 0.0)
{
  if (ok) {
    LOG_PRINT("[PASS] %s | maxAbs=%.6e maxRel=%.6e\n", name.c_str(), maxAbs, maxRel);
    g_pass++;
  } else {
    LOG_PRINT("[FAIL] %s | maxAbs=%.6e maxRel=%.6e\n", name.c_str(), maxAbs, maxRel);
    g_fail++;
  }
}

template <typename T>
static bool CreateTensor(const std::vector<T> &host,
                         const std::vector<int64_t> &shape,
                         aclDataType dt,
                         TensorHolder *holder)
{
  if (holder == nullptr) {
    return false;
  }

  const int64_t n = Numel(shape);
  if (n < 0 || n != static_cast<int64_t>(host.size())) {
    LOG_PRINT("[ERROR] CreateTensor shape/data mismatch shape=%s numel=%lld data=%zu\n",
              ShapeStr(shape).c_str(), static_cast<long long>(n), host.size());
    return false;
  }

  const size_t bytes = host.size() * sizeof(T);
  if (bytes > 0) {
    if (aclrtMalloc(&(holder->addr), bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      return false;
    }
    if (aclrtMemcpy(holder->addr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
      return false;
    }
  }

  const std::vector<int64_t> strides = MakeStrides(shape);
  const int64_t *shapePtr = shape.empty() ? nullptr : shape.data();
  const int64_t *stridePtr = strides.empty() ? nullptr : strides.data();

  holder->tensor = aclCreateTensor(shapePtr,
                                   shape.size(),
                                   dt,
                                   stridePtr,
                                   0,
                                   ACL_FORMAT_ND,
                                   shapePtr,
                                   shape.size(),
                                   holder->addr);
  return holder->tensor != nullptr;
}

template <typename T>
static bool CopyBack(const TensorHolder &holder, std::vector<T> *host)
{
  if (host == nullptr) {
    return false;
  }
  const size_t bytes = host->size() * sizeof(T);
  if (bytes == 0) {
    return true;
  }
  return aclrtMemcpy(host->data(), bytes, holder.addr, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

template <typename T>
static bool CreateScalar(T value, aclDataType dt, ScalarHolder *holder)
{
  if (holder == nullptr) {
    return false;
  }
  holder->v = aclCreateScalar(&value, dt);
  return holder->v != nullptr;
}

template <typename T>
static void ToDoubleVec(const std::vector<T> &src,
                        aclDataType dt,
                        std::vector<double> *dst)
{
  dst->resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    (*dst)[i] = HostValToDouble(src[i], dt);
  }
}

template <typename TA, typename TB>
static void BuildExpectedTensorTensor(const std::vector<int64_t> &shapeA,
                                      const std::vector<TA> &a,
                                      aclDataType dtypeA,
                                      const std::vector<int64_t> &shapeB,
                                      const std::vector<TB> &b,
                                      aclDataType dtypeB,
                                      double alpha,
                                      aclDataType outDtype,
                                      std::vector<double> *expected)
{
  std::vector<int64_t> outShape;
  BroadcastShape(shapeA, shapeB, &outShape);
  const int64_t n = Numel(outShape);
  expected->assign(static_cast<size_t>(n), 0.0);

  const std::vector<int64_t> strideA = MakeStrides(shapeA);
  const std::vector<int64_t> strideB = MakeStrides(shapeB);
  std::vector<int64_t> outIdx;

  for (int64_t i = 0; i < n; ++i) {
    UnravelIndex(i, outShape, &outIdx);
    const int64_t offA = BroadcastOffset(outIdx, shapeA, strideA);
    const int64_t offB = BroadcastOffset(outIdx, shapeB, strideB);
    const double va = HostValToDouble(a[static_cast<size_t>(offA)], dtypeA);
    const double vb = HostValToDouble(b[static_cast<size_t>(offB)], dtypeB);
    const double y = va + alpha * vb;
    (*expected)[static_cast<size_t>(i)] = QuantizeByDType(y, outDtype);
  }
}

template <typename TA>
static void BuildExpectedTensorScalar(const std::vector<int64_t> &shapeA,
                                      const std::vector<TA> &a,
                                      aclDataType dtypeA,
                                      double otherScalar,
                                      double alpha,
                                      aclDataType outDtype,
                                      std::vector<double> *expected)
{
  const int64_t n = Numel(shapeA);
  expected->assign(static_cast<size_t>(n), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    const double va = HostValToDouble(a[static_cast<size_t>(i)], dtypeA);
    const double y = va + alpha * otherScalar;
    (*expected)[static_cast<size_t>(i)] = QuantizeByDType(y, outDtype);
  }
}

template <typename TB>
static void BuildExpectedScalarTensor(double selfScalar,
                                      const std::vector<int64_t> &shapeB,
                                      const std::vector<TB> &b,
                                      aclDataType dtypeB,
                                      double alpha,
                                      aclDataType outDtype,
                                      std::vector<double> *expected)
{
  const int64_t n = Numel(shapeB);
  expected->assign(static_cast<size_t>(n), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    const double vb = HostValToDouble(b[static_cast<size_t>(i)], dtypeB);
    const double y = selfScalar + alpha * vb;
    (*expected)[static_cast<size_t>(i)] = QuantizeByDType(y, outDtype);
  }
}

template <typename TA, typename TB, typename TO, typename ALPHA_T>
static bool RunAddCase(const std::string &name,
                       const std::vector<int64_t> &shapeA,
                       const std::vector<TA> &a,
                       aclDataType dtypeA,
                       const std::vector<int64_t> &shapeB,
                       const std::vector<TB> &b,
                       aclDataType dtypeB,
                       ALPHA_T alphaValue,
                       aclDataType alphaDtype,
                       aclDataType outDtype,
                       aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("Add dta=%s dtb=%s dto=%s alphaType=%s shapeA=%s shapeB=%s\n",
            DTypeStr(dtypeA), DTypeStr(dtypeB), DTypeStr(outDtype), DTypeStr(alphaDtype),
            ShapeStr(shapeA).c_str(), ShapeStr(shapeB).c_str());

  std::vector<int64_t> outShape;
  if (!BroadcastShape(shapeA, shapeB, &outShape)) {
    Record(name, false);
    return false;
  }

  const int64_t n = Numel(outShape);
  std::vector<TO> outInit(static_cast<size_t>(n), static_cast<TO>(0));

  TensorHolder tensorA;
  TensorHolder tensorB;
  TensorHolder tensorOut;
  ScalarHolder alpha;

  if (!CreateTensor(a, shapeA, dtypeA, &tensorA) ||
      !CreateTensor(b, shapeB, dtypeB, &tensorB) ||
      !CreateTensor(outInit, outShape, outDtype, &tensorOut) ||
      !CreateScalar(alphaValue, alphaDtype, &alpha)) {
    Record(name, false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus ret = aclnnAddGetWorkspaceSize(
      tensorA.tensor, tensorB.tensor, alpha.v, tensorOut.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  WorkspaceHolder workspace;
  if (workspaceSize > 0) {
    if (aclrtMalloc(&(workspace.addr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      Record(name, false);
      return false;
    }
  }

  ret = aclnnAdd(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  std::vector<TO> got(static_cast<size_t>(n));
  if (!CopyBack(tensorOut, &got)) {
    Record(name, false);
    return false;
  }

  std::vector<double> expected;
  std::vector<double> actual;
  BuildExpectedTensorTensor(shapeA,
                            a,
                            dtypeA,
                            shapeB,
                            b,
                            dtypeB,
                            static_cast<double>(alphaValue),
                            outDtype,
                            &expected);
  ToDoubleVec(got, outDtype, &actual);

  double atol = 0.0;
  double rtol = 0.0;
  GetTolerance(outDtype, &atol, &rtol);
  CompareStats st = CompareVec(expected, actual, atol, rtol);

  if (!st.pass && st.badIndex >= 0) {
    LOG_PRINT("mismatch idx=%lld expected=%.9g got=%.9g\n",
              static_cast<long long>(st.badIndex),
              expected[static_cast<size_t>(st.badIndex)],
              actual[static_cast<size_t>(st.badIndex)]);
  }

  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

template <typename TA, typename TO, typename OTHER_T, typename ALPHA_T>
static bool RunAddsCase(const std::string &name,
                        const std::vector<int64_t> &shape,
                        const std::vector<TA> &a,
                        aclDataType dtypeA,
                        OTHER_T otherValue,
                        aclDataType otherDtype,
                        ALPHA_T alphaValue,
                        aclDataType alphaDtype,
                        aclDataType outDtype,
                        aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("Adds dta=%s dto=%s otherType=%s alphaType=%s shape=%s\n",
            DTypeStr(dtypeA), DTypeStr(outDtype), DTypeStr(otherDtype), DTypeStr(alphaDtype), ShapeStr(shape).c_str());

  const int64_t n = Numel(shape);
  std::vector<TO> outInit(static_cast<size_t>(n), static_cast<TO>(0));

  TensorHolder tensorA;
  TensorHolder tensorOut;
  ScalarHolder other;
  ScalarHolder alpha;

  if (!CreateTensor(a, shape, dtypeA, &tensorA) ||
      !CreateTensor(outInit, shape, outDtype, &tensorOut) ||
      !CreateScalar(otherValue, otherDtype, &other) ||
      !CreateScalar(alphaValue, alphaDtype, &alpha)) {
    Record(name, false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus ret = aclnnAddsGetWorkspaceSize(
      tensorA.tensor, other.v, alpha.v, tensorOut.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  WorkspaceHolder workspace;
  if (workspaceSize > 0) {
    if (aclrtMalloc(&(workspace.addr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      Record(name, false);
      return false;
    }
  }

  ret = aclnnAdds(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  std::vector<TO> got(static_cast<size_t>(n));
  if (!CopyBack(tensorOut, &got)) {
    Record(name, false);
    return false;
  }

  std::vector<double> expected;
  std::vector<double> actual;
  BuildExpectedTensorScalar(shape,
                            a,
                            dtypeA,
                            static_cast<double>(otherValue),
                            static_cast<double>(alphaValue),
                            outDtype,
                            &expected);
  ToDoubleVec(got, outDtype, &actual);

  double atol = 0.0;
  double rtol = 0.0;
  GetTolerance(outDtype, &atol, &rtol);
  CompareStats st = CompareVec(expected, actual, atol, rtol);

  if (!st.pass && st.badIndex >= 0) {
    LOG_PRINT("mismatch idx=%lld expected=%.9g got=%.9g\n",
              static_cast<long long>(st.badIndex),
              expected[static_cast<size_t>(st.badIndex)],
              actual[static_cast<size_t>(st.badIndex)]);
  }

  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

template <typename TA, typename TO, typename OTHER_T, typename ALPHA_T>
static bool RunAddsStatusOnlyCase(const std::string &name,
                                  const std::vector<int64_t> &shape,
                                  const std::vector<TA> &a,
                                  aclDataType dtypeA,
                                  OTHER_T otherValue,
                                  aclDataType otherDtype,
                                  ALPHA_T alphaValue,
                                  aclDataType alphaDtype,
                                  aclDataType outDtype,
                                  aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("Adds(StatusOnly) dta=%s dto=%s otherType=%s alphaType=%s shape=%s\n",
            DTypeStr(dtypeA), DTypeStr(outDtype), DTypeStr(otherDtype), DTypeStr(alphaDtype), ShapeStr(shape).c_str());

  const int64_t n = Numel(shape);
  std::vector<TO> outInit(static_cast<size_t>(n), static_cast<TO>(0));

  TensorHolder tensorA;
  TensorHolder tensorOut;
  ScalarHolder other;
  ScalarHolder alpha;

  if (!CreateTensor(a, shape, dtypeA, &tensorA) ||
      !CreateTensor(outInit, shape, outDtype, &tensorOut) ||
      !CreateScalar(otherValue, otherDtype, &other) ||
      !CreateScalar(alphaValue, alphaDtype, &alpha)) {
    Record(name, false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus ret = aclnnAddsGetWorkspaceSize(
      tensorA.tensor, other.v, alpha.v, tensorOut.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  WorkspaceHolder workspace;
  if (workspaceSize > 0) {
    if (aclrtMalloc(&(workspace.addr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      Record(name, false);
      return false;
    }
  }

  ret = aclnnAdds(workspace.addr, workspaceSize, executor, stream);
  const bool ok = (ret == ACL_SUCCESS) && (aclrtSynchronizeStream(stream) == ACL_SUCCESS);
  Record(name, ok);
  return ok;
}

template <typename TA, typename TB, typename ALPHA_T>
static bool RunInplaceAddCase(const std::string &name,
                              const std::vector<int64_t> &shapeA,
                              const std::vector<TA> &a,
                              aclDataType dtypeA,
                              const std::vector<int64_t> &shapeB,
                              const std::vector<TB> &b,
                              aclDataType dtypeB,
                              ALPHA_T alphaValue,
                              aclDataType alphaDtype,
                              aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("InplaceAdd dta=%s dtb=%s alphaType=%s shapeA=%s shapeB=%s\n",
            DTypeStr(dtypeA), DTypeStr(dtypeB), DTypeStr(alphaDtype), ShapeStr(shapeA).c_str(), ShapeStr(shapeB).c_str());

  std::vector<int64_t> broadcastShape;
  if (!BroadcastShape(shapeA, shapeB, &broadcastShape) || broadcastShape != shapeA) {
    Record(name, false);
    return false;
  }

  TensorHolder tensorA;
  TensorHolder tensorB;
  ScalarHolder alpha;

  if (!CreateTensor(a, shapeA, dtypeA, &tensorA) ||
      !CreateTensor(b, shapeB, dtypeB, &tensorB) ||
      !CreateScalar(alphaValue, alphaDtype, &alpha)) {
    Record(name, false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus ret = aclnnInplaceAddGetWorkspaceSize(
      tensorA.tensor, tensorB.tensor, alpha.v, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  WorkspaceHolder workspace;
  if (workspaceSize > 0) {
    if (aclrtMalloc(&(workspace.addr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      Record(name, false);
      return false;
    }
  }

  ret = aclnnInplaceAdd(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  const int64_t n = Numel(shapeA);
  std::vector<TA> got(static_cast<size_t>(n));
  if (!CopyBack(tensorA, &got)) {
    Record(name, false);
    return false;
  }

  std::vector<double> expected;
  std::vector<double> actual;
  BuildExpectedTensorTensor(shapeA,
                            a,
                            dtypeA,
                            shapeB,
                            b,
                            dtypeB,
                            static_cast<double>(alphaValue),
                            dtypeA,
                            &expected);
  ToDoubleVec(got, dtypeA, &actual);

  double atol = 0.0;
  double rtol = 0.0;
  GetTolerance(dtypeA, &atol, &rtol);
  CompareStats st = CompareVec(expected, actual, atol, rtol);

  if (!st.pass && st.badIndex >= 0) {
    LOG_PRINT("mismatch idx=%lld expected=%.9g got=%.9g\n",
              static_cast<long long>(st.badIndex),
              expected[static_cast<size_t>(st.badIndex)],
              actual[static_cast<size_t>(st.badIndex)]);
  }

  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

template <typename TA, typename OTHER_T, typename ALPHA_T>
static bool RunInplaceAddsCase(const std::string &name,
                               const std::vector<int64_t> &shape,
                               const std::vector<TA> &a,
                               aclDataType dtypeA,
                               OTHER_T otherValue,
                               aclDataType otherDtype,
                               ALPHA_T alphaValue,
                               aclDataType alphaDtype,
                               aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("InplaceAdds dta=%s otherType=%s alphaType=%s shape=%s\n",
            DTypeStr(dtypeA), DTypeStr(otherDtype), DTypeStr(alphaDtype), ShapeStr(shape).c_str());

  TensorHolder tensorA;
  ScalarHolder other;
  ScalarHolder alpha;

  if (!CreateTensor(a, shape, dtypeA, &tensorA) ||
      !CreateScalar(otherValue, otherDtype, &other) ||
      !CreateScalar(alphaValue, alphaDtype, &alpha)) {
    Record(name, false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus ret = aclnnInplaceAddsGetWorkspaceSize(
      tensorA.tensor, other.v, alpha.v, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  WorkspaceHolder workspace;
  if (workspaceSize > 0) {
    if (aclrtMalloc(&(workspace.addr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      Record(name, false);
      return false;
    }
  }

  ret = aclnnInplaceAdds(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  const int64_t n = Numel(shape);
  std::vector<TA> got(static_cast<size_t>(n));
  if (!CopyBack(tensorA, &got)) {
    Record(name, false);
    return false;
  }

  std::vector<double> expected;
  std::vector<double> actual;
  BuildExpectedTensorScalar(shape,
                            a,
                            dtypeA,
                            static_cast<double>(otherValue),
                            static_cast<double>(alphaValue),
                            dtypeA,
                            &expected);
  ToDoubleVec(got, dtypeA, &actual);

  double atol = 0.0;
  double rtol = 0.0;
  GetTolerance(dtypeA, &atol, &rtol);
  CompareStats st = CompareVec(expected, actual, atol, rtol);

  if (!st.pass && st.badIndex >= 0) {
    LOG_PRINT("mismatch idx=%lld expected=%.9g got=%.9g\n",
              static_cast<long long>(st.badIndex),
              expected[static_cast<size_t>(st.badIndex)],
              actual[static_cast<size_t>(st.badIndex)]);
  }

  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

template <typename TB, typename TO, typename SELF_T, typename ALPHA_T>
static bool RunAddV3Case(const std::string &name,
                         SELF_T selfValue,
                         aclDataType selfDtype,
                         const std::vector<int64_t> &shapeB,
                         const std::vector<TB> &b,
                         aclDataType dtypeB,
                         ALPHA_T alphaValue,
                         aclDataType alphaDtype,
                         aclDataType outDtype,
                         aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("AddV3 selfType=%s dtb=%s dto=%s alphaType=%s shapeB=%s\n",
            DTypeStr(selfDtype), DTypeStr(dtypeB), DTypeStr(outDtype), DTypeStr(alphaDtype), ShapeStr(shapeB).c_str());

  const int64_t n = Numel(shapeB);
  std::vector<TO> outInit(static_cast<size_t>(n), static_cast<TO>(0));

  TensorHolder tensorB;
  TensorHolder tensorOut;
  ScalarHolder self;
  ScalarHolder alpha;

  if (!CreateTensor(b, shapeB, dtypeB, &tensorB) ||
      !CreateTensor(outInit, shapeB, outDtype, &tensorOut) ||
      !CreateScalar(selfValue, selfDtype, &self) ||
      !CreateScalar(alphaValue, alphaDtype, &alpha)) {
    Record(name, false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus ret = aclnnAddV3GetWorkspaceSize(
      self.v, tensorB.tensor, alpha.v, tensorOut.tensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  WorkspaceHolder workspace;
  if (workspaceSize > 0) {
    if (aclrtMalloc(&(workspace.addr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      Record(name, false);
      return false;
    }
  }

  ret = aclnnAddV3(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  std::vector<TO> got(static_cast<size_t>(n));
  if (!CopyBack(tensorOut, &got)) {
    Record(name, false);
    return false;
  }

  std::vector<double> expected;
  std::vector<double> actual;
  BuildExpectedScalarTensor(static_cast<double>(selfValue),
                            shapeB,
                            b,
                            dtypeB,
                            static_cast<double>(alphaValue),
                            outDtype,
                            &expected);
  ToDoubleVec(got, outDtype, &actual);

  double atol = 0.0;
  double rtol = 0.0;
  GetTolerance(outDtype, &atol, &rtol);
  CompareStats st = CompareVec(expected, actual, atol, rtol);

  if (!st.pass && st.badIndex >= 0) {
    LOG_PRINT("mismatch idx=%lld expected=%.9g got=%.9g\n",
              static_cast<long long>(st.badIndex),
              expected[static_cast<size_t>(st.badIndex)],
              actual[static_cast<size_t>(st.badIndex)]);
  }

  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

template <typename TB, typename SELF_T, typename ALPHA_T>
static bool RunInplaceAddV3Case(const std::string &name,
                                SELF_T selfValue,
                                aclDataType selfDtype,
                                const std::vector<int64_t> &shapeB,
                                const std::vector<TB> &b,
                                aclDataType dtypeB,
                                ALPHA_T alphaValue,
                                aclDataType alphaDtype,
                                aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("InplaceAddV3 selfType=%s dtb=%s alphaType=%s shapeB=%s\n",
            DTypeStr(selfDtype), DTypeStr(dtypeB), DTypeStr(alphaDtype), ShapeStr(shapeB).c_str());

  TensorHolder tensorB;
  ScalarHolder self;
  ScalarHolder alpha;

  if (!CreateTensor(b, shapeB, dtypeB, &tensorB) ||
      !CreateScalar(selfValue, selfDtype, &self) ||
      !CreateScalar(alphaValue, alphaDtype, &alpha)) {
    Record(name, false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus ret = aclnnInplaceAddV3GetWorkspaceSize(
      self.v, tensorB.tensor, alpha.v, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  WorkspaceHolder workspace;
  if (workspaceSize > 0) {
    if (aclrtMalloc(&(workspace.addr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      Record(name, false);
      return false;
    }
  }

  ret = aclnnInplaceAddV3(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  const int64_t n = Numel(shapeB);
  std::vector<TB> got(static_cast<size_t>(n));
  if (!CopyBack(tensorB, &got)) {
    Record(name, false);
    return false;
  }

  std::vector<double> expected;
  std::vector<double> actual;
  BuildExpectedScalarTensor(static_cast<double>(selfValue),
                            shapeB,
                            b,
                            dtypeB,
                            static_cast<double>(alphaValue),
                            dtypeB,
                            &expected);
  ToDoubleVec(got, dtypeB, &actual);

  double atol = 0.0;
  double rtol = 0.0;
  GetTolerance(dtypeB, &atol, &rtol);
  CompareStats st = CompareVec(expected, actual, atol, rtol);

  if (!st.pass && st.badIndex >= 0) {
    LOG_PRINT("mismatch idx=%lld expected=%.9g got=%.9g\n",
              static_cast<long long>(st.badIndex),
              expected[static_cast<size_t>(st.badIndex)],
              actual[static_cast<size_t>(st.badIndex)]);
  }

  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

static bool RunExpectedFail_GetWorkspace(const std::string &name, aclnnStatus ret)
{
  const bool ok = (ret != ACL_SUCCESS);
  Record(name, ok);
  return ok;
}

static bool InitAcl(int32_t deviceId, aclrtStream *stream)
{
  aclError ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclInit failed: %d\n", ret);
    return false;
  }

  ret = aclrtSetDevice(deviceId);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclrtSetDevice failed: %d\n", ret);
    return false;
  }

  ret = aclrtCreateStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclrtCreateStream failed: %d\n", ret);
    return false;
  }

  return true;
}

static void FinalizeAcl(int32_t deviceId, aclrtStream stream)
{
  if (stream != nullptr) {
    aclrtDestroyStream(stream);
  }
  aclrtResetDevice(deviceId);
  aclFinalize();
}

static std::vector<float> RepeatFloat(const std::vector<float> &pattern, int64_t n)
{
  std::vector<float> out;
  if (pattern.empty() || n <= 0) {
    return out;
  }
  out.resize(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] = pattern[static_cast<size_t>(i % static_cast<int64_t>(pattern.size()))];
  }
  return out;
}

int main()
{
  const int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  if (!InitAcl(deviceId, &stream)) {
    return -1;
  }

  LOG_PRINT("========================================\n");
  LOG_PRINT(" Add Score Test Suite (Self-contained)\n");
  LOG_PRINT("========================================\n");

  // 1) Add: 基础和广播覆盖
  RunAddCase<float, float, float, float>(
      "Add_Fp32_Basic_Alpha1", {2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT,
      {2, 3}, {10, 20, 30, 40, 50, 60}, ACL_FLOAT,
      1.0f, ACL_FLOAT, ACL_FLOAT, stream);

  RunAddCase<float, float, float, float>(
      "Add_Fp32_Broadcast_AlphaNeg", {4, 3},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, ACL_FLOAT,
      {3}, {2, 4, 8}, ACL_FLOAT,
      -0.5f, ACL_FLOAT, ACL_FLOAT, stream);

  std::vector<uint16_t> fp16A = {F32ToF16(1.0f), F32ToF16(2.0f), F32ToF16(3.0f), F32ToF16(-4.0f)};
  std::vector<uint16_t> fp16B = {F32ToF16(0.5f), F32ToF16(-1.5f), F32ToF16(2.0f), F32ToF16(3.0f)};
  RunAddCase<uint16_t, uint16_t, uint16_t, float>(
      "Add_Fp16_Basic_Alpha1", {2, 2}, fp16A, ACL_FLOAT16,
      {2, 2}, fp16B, ACL_FLOAT16,
      1.0f, ACL_FLOAT, ACL_FLOAT16, stream);

  std::vector<uint16_t> bf16A = {F32ToBF16(1.0f), F32ToBF16(2.0f), F32ToBF16(3.0f), F32ToBF16(4.0f)};
  std::vector<uint16_t> bf16B = {F32ToBF16(-0.25f), F32ToBF16(1.25f), F32ToBF16(1.5f), F32ToBF16(-2.0f)};
  RunAddCase<uint16_t, uint16_t, uint16_t, float>(
      "Add_Bf16_Basic_AlphaFloat", {2, 2}, bf16A, ACL_BF16,
      {2, 2}, bf16B, ACL_BF16,
      1.25f, ACL_FLOAT, ACL_BF16, stream);

  RunAddCase<int32_t, int32_t, int32_t, int32_t>(
      "Add_Int32_AlphaNeg", {2, 3}, {1, -2, 3, -4, 5, -6}, ACL_INT32,
      {2, 3}, {7, 8, -9, 10, -11, 12}, ACL_INT32,
      -2, ACL_INT32, ACL_INT32, stream);

  RunAddCase<int64_t, int64_t, int64_t, int64_t>(
      "Add_Int64_Alpha2", {2, 2}, {10, 20, -30, 40}, ACL_INT64,
      {2, 2}, {1, 2, 3, 4}, ACL_INT64,
      2, ACL_INT64, ACL_INT64, stream);

  RunAddCase<int8_t, int8_t, int8_t, int32_t>(
      "Add_Int8_Alpha2", {2, 3},
      {static_cast<int8_t>(1), static_cast<int8_t>(2), static_cast<int8_t>(3),
       static_cast<int8_t>(-4), static_cast<int8_t>(-5), static_cast<int8_t>(6)}, ACL_INT8,
      {2, 3},
      {static_cast<int8_t>(-1), static_cast<int8_t>(1), static_cast<int8_t>(2),
       static_cast<int8_t>(3), static_cast<int8_t>(-2), static_cast<int8_t>(-3)}, ACL_INT8,
      2, ACL_INT32, ACL_INT8, stream);

  RunAddCase<uint16_t, float, float, float>(
      "Add_Mix_Fp16_Fp32_Alpha1", {2, 2}, fp16A, ACL_FLOAT16,
      {2, 2}, {0.25f, -0.5f, 1.5f, 3.0f}, ACL_FLOAT,
      1.0f, ACL_FLOAT, ACL_FLOAT, stream);

  RunAddCase<uint16_t, float, float, float>(
      "Add_Mix_Bf16_Fp32_Alpha1", {2, 2}, bf16A, ACL_BF16,
      {2, 2}, {-1.0f, 0.5f, 2.0f, -0.25f}, ACL_FLOAT,
      1.0f, ACL_FLOAT, ACL_FLOAT, stream);

  RunAddCase<float, float, float, float>(
      "Add_EmptyTensor", {0, 4}, {}, ACL_FLOAT,
      {0, 4}, {}, ACL_FLOAT,
      1.0f, ACL_FLOAT, ACL_FLOAT, stream);

  std::vector<float> largeA = RepeatFloat({-2.0f, -1.0f, 0.25f, 1.5f, 3.0f}, 512 * 256);
  std::vector<float> largeB = RepeatFloat({1.0f, -0.5f, 2.0f}, 512 * 256);
  RunAddCase<float, float, float, float>(
      "Add_Fp32_Large_512x256", {512, 256}, largeA, ACL_FLOAT,
      {512, 256}, largeB, ACL_FLOAT,
      0.75f, ACL_FLOAT, ACL_FLOAT, stream);

  // 2) Adds: Tensor + scalar
  RunAddsCase<float, float, float, float>(
      "Adds_Fp32_Scalar_Alpha1", {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT,
      2.5f, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      ACL_FLOAT, stream);

  RunAddsCase<float, float, float, float>(
      "Adds_Fp32_Scalar_AlphaNeg", {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT,
      -3.0f, ACL_FLOAT,
      -0.5f, ACL_FLOAT,
      ACL_FLOAT, stream);

  RunAddsCase<int32_t, int32_t, int32_t, int32_t>(
      "Adds_Int32_Scalar", {2, 3}, {1, 2, 3, -4, -5, 6}, ACL_INT32,
      3, ACL_INT32,
      2, ACL_INT32,
      ACL_INT32, stream);

  RunAddsCase<uint16_t, uint16_t, float, float>(
      "Adds_Fp16_Scalar", {2, 2}, fp16A, ACL_FLOAT16,
      1.5f, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      ACL_FLOAT16, stream);

  // bool + bool + bool 且 out=uint8，触发 aclnnAdds 中 bool 特殊分支（只校验运行成功）
  std::vector<uint8_t> boolIn = {0, 1, 0, 1, 1, 0};
  RunAddsStatusOnlyCase<uint8_t, uint8_t, uint8_t, uint8_t>(
      "Adds_Bool_SpecialBranch", {2, 3}, boolIn, ACL_BOOL,
      static_cast<uint8_t>(1), ACL_BOOL,
      static_cast<uint8_t>(1), ACL_BOOL,
      ACL_UINT8, stream);

  RunAddsCase<float, float, float, float>(
      "Adds_EmptyTensor", {0, 8}, {}, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      2.0f, ACL_FLOAT,
      ACL_FLOAT, stream);

  // 3) InplaceAdd / InplaceAdds
  RunInplaceAddCase<float, float, float>(
      "InplaceAdd_Fp32_Broadcast", {2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT,
      {3}, {10, 20, 30}, ACL_FLOAT,
      1.0f, ACL_FLOAT, stream);

  RunInplaceAddCase<int64_t, int64_t, int64_t>(
      "InplaceAdd_Int64_Alpha2", {2, 2}, {10, -20, 30, -40}, ACL_INT64,
      {2, 2}, {1, 2, 3, 4}, ACL_INT64,
      2, ACL_INT64, stream);

  RunInplaceAddCase<uint16_t, uint16_t, float>(
      "InplaceAdd_Fp16_AlphaNeg", {2, 2}, fp16A, ACL_FLOAT16,
      {2, 2}, fp16B, ACL_FLOAT16,
      -1.0f, ACL_FLOAT, stream);

  RunInplaceAddsCase<float, float, float>(
      "InplaceAdds_Fp32_Scalar", {2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT,
      2.0f, ACL_FLOAT,
      1.5f, ACL_FLOAT,
      stream);

  RunInplaceAddsCase<int32_t, int32_t, int32_t>(
      "InplaceAdds_Int32_Scalar", {2, 3}, {1, -2, 3, -4, 5, -6}, ACL_INT32,
      2, ACL_INT32,
      -3, ACL_INT32,
      stream);

  // 4) AddV3 / InplaceAddV3
  RunAddV3Case<float, float, float, float>(
      "AddV3_Fp32_Alpha1", 3.0f, ACL_FLOAT,
      {2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      ACL_FLOAT, stream);

  RunAddV3Case<float, float, float, float>(
      "AddV3_Fp32_AlphaNeg", -2.0f, ACL_FLOAT,
      {2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT,
      -0.5f, ACL_FLOAT,
      ACL_FLOAT, stream);

  RunAddV3Case<int8_t, int8_t, int32_t, int32_t>(
      "AddV3_Int8_Alpha2", 2, ACL_INT32,
      {2, 3},
      {static_cast<int8_t>(1), static_cast<int8_t>(2), static_cast<int8_t>(3),
       static_cast<int8_t>(-4), static_cast<int8_t>(5), static_cast<int8_t>(-6)}, ACL_INT8,
      2, ACL_INT32,
      ACL_INT8, stream);

  RunInplaceAddV3Case<float, float, float>(
      "InplaceAddV3_Fp32_Alpha1", 2.0f, ACL_FLOAT,
      {2, 2}, {1, 2, 3, 4}, ACL_FLOAT,
      1.0f, ACL_FLOAT,
      stream);

  RunInplaceAddV3Case<int8_t, int32_t, int32_t>(
      "InplaceAddV3_Int8_Alpha2", -1, ACL_INT32,
      {2, 3},
      {static_cast<int8_t>(2), static_cast<int8_t>(-3), static_cast<int8_t>(4),
       static_cast<int8_t>(1), static_cast<int8_t>(2), static_cast<int8_t>(-2)}, ACL_INT8,
      2, ACL_INT32,
      stream);

  // 5) 参数异常分支（只校验 GetWorkspace 返回错误）
  {
    TensorHolder t1;
    TensorHolder t2;
    TensorHolder tout;
    ScalarHolder alpha;
    CreateTensor(std::vector<float>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &t1);
    CreateTensor(std::vector<float>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &t2);
    CreateTensor(std::vector<float>{0, 0, 0, 0}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &tout);
    CreateScalar(1.0f, ACL_FLOAT, &alpha);

    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    RunExpectedFail_GetWorkspace("Add_NullSelf_FailExpected",
                                 aclnnAddGetWorkspaceSize(nullptr, t2.tensor, alpha.v, tout.tensor, &ws, &ex));
    RunExpectedFail_GetWorkspace("Add_NullOther_FailExpected",
                                 aclnnAddGetWorkspaceSize(t1.tensor, nullptr, alpha.v, tout.tensor, &ws, &ex));
    RunExpectedFail_GetWorkspace("Add_NullAlpha_FailExpected",
                                 aclnnAddGetWorkspaceSize(t1.tensor, t2.tensor, nullptr, tout.tensor, &ws, &ex));
    RunExpectedFail_GetWorkspace("Add_NullOut_FailExpected",
                                 aclnnAddGetWorkspaceSize(t1.tensor, t2.tensor, alpha.v, nullptr, &ws, &ex));
  }

  {
    TensorHolder t1;
    TensorHolder t2;
    TensorHolder toutBad;
    ScalarHolder alpha;
    CreateTensor(std::vector<float>{1, 2, 3, 4, 5, 6}, std::vector<int64_t>{2, 3}, ACL_FLOAT, &t1);
    CreateTensor(std::vector<float>{1, 2, 3}, std::vector<int64_t>{3}, ACL_FLOAT, &t2);
    CreateTensor(std::vector<float>{0, 0, 0, 0}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &toutBad);
    CreateScalar(1.0f, ACL_FLOAT, &alpha);

    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    RunExpectedFail_GetWorkspace("Add_InvalidOutShape_FailExpected",
                                 aclnnAddGetWorkspaceSize(t1.tensor, t2.tensor, alpha.v, toutBad.tensor, &ws, &ex));
  }

  {
    TensorHolder t1;
    TensorHolder t2;
    TensorHolder tout;
    ScalarHolder alpha;
    CreateTensor(std::vector<float>(6, 1.0f), std::vector<int64_t>{2, 3}, ACL_FLOAT, &t1);
    CreateTensor(std::vector<float>(20, 1.0f), std::vector<int64_t>{4, 5}, ACL_FLOAT, &t2);
    CreateTensor(std::vector<float>(6, 0.0f), std::vector<int64_t>{2, 3}, ACL_FLOAT, &tout);
    CreateScalar(1.0f, ACL_FLOAT, &alpha);

    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    RunExpectedFail_GetWorkspace("Add_InvalidBroadcast_FailExpected",
                                 aclnnAddGetWorkspaceSize(t1.tensor, t2.tensor, alpha.v, tout.tensor, &ws, &ex));
  }

  {
    TensorHolder t;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;
    CreateTensor(std::vector<float>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &t);
    CreateTensor(std::vector<float>{0, 0, 0, 0}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &out);
    CreateScalar(1.0f, ACL_FLOAT, &other);
    CreateScalar(1.0f, ACL_FLOAT, &alpha);

    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    RunExpectedFail_GetWorkspace("Adds_NullSelf_FailExpected",
                                 aclnnAddsGetWorkspaceSize(nullptr, other.v, alpha.v, out.tensor, &ws, &ex));
    RunExpectedFail_GetWorkspace("Adds_NullOther_FailExpected",
                                 aclnnAddsGetWorkspaceSize(t.tensor, nullptr, alpha.v, out.tensor, &ws, &ex));
    RunExpectedFail_GetWorkspace("Adds_NullAlpha_FailExpected",
                                 aclnnAddsGetWorkspaceSize(t.tensor, other.v, nullptr, out.tensor, &ws, &ex));
  }

  {
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha;
    CreateTensor(std::vector<float>{1, 2}, std::vector<int64_t>{2, 1}, ACL_FLOAT, &self);
    CreateTensor(std::vector<float>{1, 2, 3, 4, 5, 6}, std::vector<int64_t>{2, 3}, ACL_FLOAT, &other);
    CreateScalar(1.0f, ACL_FLOAT, &alpha);

    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    RunExpectedFail_GetWorkspace("InplaceAdd_SelfShapeMismatch_FailExpected",
                                 aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.v, &ws, &ex));
  }

  {
    TensorHolder self;
    ScalarHolder other;
    ScalarHolder alpha;
    CreateTensor(std::vector<float>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &self);
    CreateScalar(1.0f, ACL_FLOAT, &other);
    CreateScalar(1.0f, ACL_FLOAT, &alpha);

    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    RunExpectedFail_GetWorkspace("InplaceAdds_NullSelf_FailExpected",
                                 aclnnInplaceAddsGetWorkspaceSize(nullptr, other.v, alpha.v, &ws, &ex));
  }

  {
    TensorHolder other;
    TensorHolder out;
    ScalarHolder self;
    ScalarHolder alpha;
    CreateTensor(std::vector<float>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &other);
    CreateTensor(std::vector<float>{0, 0, 0, 0}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &out);
    CreateScalar(1.0f, ACL_FLOAT, &self);
    CreateScalar(1.0f, ACL_FLOAT, &alpha);

    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    RunExpectedFail_GetWorkspace("AddV3_NullSelf_FailExpected",
                                 aclnnAddV3GetWorkspaceSize(nullptr, other.tensor, alpha.v, out.tensor, &ws, &ex));
  }

  {
    TensorHolder other;
    TensorHolder outBad;
    ScalarHolder self;
    ScalarHolder alpha;
    CreateTensor(std::vector<float>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &other);
    CreateTensor(std::vector<float>{0, 0, 0}, std::vector<int64_t>{3}, ACL_FLOAT, &outBad);
    CreateScalar(1.0f, ACL_FLOAT, &self);
    CreateScalar(1.0f, ACL_FLOAT, &alpha);

    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    RunExpectedFail_GetWorkspace("AddV3_OutShapeMismatch_FailExpected",
                                 aclnnAddV3GetWorkspaceSize(self.v, other.tensor, alpha.v, outBad.tensor, &ws, &ex));
  }

  LOG_PRINT("\n========================================\n");
  LOG_PRINT("Total=%d Passed=%d Failed=%d\n", g_pass + g_fail, g_pass, g_fail);
  LOG_PRINT("========================================\n");

  FinalizeAcl(deviceId, stream);
  return (g_fail == 0) ? 0 : 1;
}
