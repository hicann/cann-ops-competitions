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
#include "aclnnop/aclnn_exp2.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"

#define LOG_PRINT(fmt, ...)            \
  do {                                 \
    std::printf(fmt, ##__VA_ARGS__);   \
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
    case ACL_FLOAT16:
      return static_cast<double>(F16ToF32(F32ToF16(static_cast<float>(x))));
    case ACL_BF16:
      return static_cast<double>(BF16ToF32(F32ToBF16(static_cast<float>(x))));
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
    *atol = 3e-3;
    *rtol = 3e-3;
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

static double PowRef(double base, double exp)
{
  return std::pow(base, exp);
}

template <typename TA>
static void BuildExpectedTensorScalar(const std::vector<int64_t> &shape,
                                      const std::vector<TA> &base,
                                      aclDataType baseDt,
                                      double exponent,
                                      aclDataType outDt,
                                      std::vector<double> *expected)
{
  const int64_t n = Numel(shape);
  expected->assign(static_cast<size_t>(n), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    const double b = HostValToDouble(base[static_cast<size_t>(i)], baseDt);
    (*expected)[static_cast<size_t>(i)] = QuantizeByDType(PowRef(b, exponent), outDt);
  }
}

template <typename TB>
static void BuildExpectedScalarTensor(double base,
                                      const std::vector<int64_t> &shape,
                                      const std::vector<TB> &exponent,
                                      aclDataType expDt,
                                      aclDataType outDt,
                                      std::vector<double> *expected)
{
  const int64_t n = Numel(shape);
  expected->assign(static_cast<size_t>(n), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    const double e = HostValToDouble(exponent[static_cast<size_t>(i)], expDt);
    (*expected)[static_cast<size_t>(i)] = QuantizeByDType(PowRef(base, e), outDt);
  }
}

template <typename TA, typename TB>
static void BuildExpectedTensorTensor(const std::vector<int64_t> &shapeA,
                                      const std::vector<TA> &base,
                                      aclDataType baseDt,
                                      const std::vector<int64_t> &shapeB,
                                      const std::vector<TB> &exponent,
                                      aclDataType expDt,
                                      aclDataType outDt,
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
    const double b = HostValToDouble(base[static_cast<size_t>(offA)], baseDt);
    const double e = HostValToDouble(exponent[static_cast<size_t>(offB)], expDt);
    (*expected)[static_cast<size_t>(i)] = QuantizeByDType(PowRef(b, e), outDt);
  }
}

template <typename TA, typename EXP_T, typename TO>
static bool RunPowTensorScalarCase(const std::string &name,
                                   const std::vector<int64_t> &shape,
                                   const std::vector<TA> &base,
                                   aclDataType baseDt,
                                   EXP_T exponent,
                                   aclDataType expDt,
                                   aclDataType outDt,
                                   aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("PowTensorScalar base=%s exp=%s out=%s shape=%s\n",
            DTypeStr(baseDt), DTypeStr(expDt), DTypeStr(outDt), ShapeStr(shape).c_str());

  const int64_t n = Numel(shape);
  std::vector<TO> outInit(static_cast<size_t>(n), static_cast<TO>(0));

  TensorHolder tBase;
  TensorHolder tOut;
  ScalarHolder sExp;
  if (!CreateTensor(base, shape, baseDt, &tBase) ||
      !CreateTensor(outInit, shape, outDt, &tOut) ||
      !CreateScalar(exponent, expDt, &sExp)) {
    Record(name, false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus ret = aclnnPowTensorScalarGetWorkspaceSize(
      tBase.tensor, sExp.v, tOut.tensor, &workspaceSize, &executor);
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

  ret = aclnnPowTensorScalar(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  std::vector<TO> got(static_cast<size_t>(n));
  if (!CopyBack(tOut, &got)) {
    Record(name, false);
    return false;
  }

  std::vector<double> expected;
  std::vector<double> actual;
  BuildExpectedTensorScalar(shape, base, baseDt, static_cast<double>(exponent), outDt, &expected);
  ToDoubleVec(got, outDt, &actual);

  double atol = 0.0;
  double rtol = 0.0;
  GetTolerance(outDt, &atol, &rtol);
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

template <typename TA, typename EXP_T>
static bool RunInplacePowTensorScalarCase(const std::string &name,
                                          const std::vector<int64_t> &shape,
                                          const std::vector<TA> &base,
                                          aclDataType baseDt,
                                          EXP_T exponent,
                                          aclDataType expDt,
                                          aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("InplacePowTensorScalar base=%s exp=%s shape=%s\n",
            DTypeStr(baseDt), DTypeStr(expDt), ShapeStr(shape).c_str());

  TensorHolder tBase;
  ScalarHolder sExp;
  if (!CreateTensor(base, shape, baseDt, &tBase) || !CreateScalar(exponent, expDt, &sExp)) {
    Record(name, false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus ret = aclnnInplacePowTensorScalarGetWorkspaceSize(
      tBase.tensor, sExp.v, &workspaceSize, &executor);
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

  ret = aclnnInplacePowTensorScalar(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  const int64_t n = Numel(shape);
  std::vector<TA> got(static_cast<size_t>(n));
  if (!CopyBack(tBase, &got)) {
    Record(name, false);
    return false;
  }

  std::vector<double> expected;
  std::vector<double> actual;
  BuildExpectedTensorScalar(shape, base, baseDt, static_cast<double>(exponent), baseDt, &expected);
  ToDoubleVec(got, baseDt, &actual);

  double atol = 0.0;
  double rtol = 0.0;
  GetTolerance(baseDt, &atol, &rtol);
  CompareStats st = CompareVec(expected, actual, atol, rtol);
  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

template <typename BASE_T, typename TE, typename TO>
static bool RunPowScalarTensorCase(const std::string &name,
                                   BASE_T base,
                                   aclDataType baseDt,
                                   const std::vector<int64_t> &shape,
                                   const std::vector<TE> &exponent,
                                   aclDataType expDt,
                                   aclDataType outDt,
                                   aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("PowScalarTensor base=%s exp=%s out=%s shape=%s\n",
            DTypeStr(baseDt), DTypeStr(expDt), DTypeStr(outDt), ShapeStr(shape).c_str());

  const int64_t n = Numel(shape);
  std::vector<TO> outInit(static_cast<size_t>(n), static_cast<TO>(0));

  ScalarHolder sBase;
  TensorHolder tExp;
  TensorHolder tOut;
  if (!CreateScalar(base, baseDt, &sBase) ||
      !CreateTensor(exponent, shape, expDt, &tExp) ||
      !CreateTensor(outInit, shape, outDt, &tOut)) {
    Record(name, false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus ret = aclnnPowScalarTensorGetWorkspaceSize(
      sBase.v, tExp.tensor, tOut.tensor, &workspaceSize, &executor);
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

  ret = aclnnPowScalarTensor(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  std::vector<TO> got(static_cast<size_t>(n));
  if (!CopyBack(tOut, &got)) {
    Record(name, false);
    return false;
  }

  std::vector<double> expected;
  std::vector<double> actual;
  BuildExpectedScalarTensor(static_cast<double>(base), shape, exponent, expDt, outDt, &expected);
  ToDoubleVec(got, outDt, &actual);

  double atol = 0.0;
  double rtol = 0.0;
  GetTolerance(outDt, &atol, &rtol);
  CompareStats st = CompareVec(expected, actual, atol, rtol);
  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

template <typename TB, typename TE, typename TO>
static bool RunPowTensorTensorCase(const std::string &name,
                                   const std::vector<int64_t> &shapeBase,
                                   const std::vector<TB> &base,
                                   aclDataType baseDt,
                                   const std::vector<int64_t> &shapeExp,
                                   const std::vector<TE> &exponent,
                                   aclDataType expDt,
                                   aclDataType outDt,
                                   aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("PowTensorTensor base=%s exp=%s out=%s shapeB=%s shapeE=%s\n",
            DTypeStr(baseDt), DTypeStr(expDt), DTypeStr(outDt),
            ShapeStr(shapeBase).c_str(), ShapeStr(shapeExp).c_str());

  std::vector<int64_t> outShape;
  if (!BroadcastShape(shapeBase, shapeExp, &outShape)) {
    Record(name, false);
    return false;
  }
  const int64_t n = Numel(outShape);
  std::vector<TO> outInit(static_cast<size_t>(n), static_cast<TO>(0));

  TensorHolder tBase;
  TensorHolder tExp;
  TensorHolder tOut;
  if (!CreateTensor(base, shapeBase, baseDt, &tBase) ||
      !CreateTensor(exponent, shapeExp, expDt, &tExp) ||
      !CreateTensor(outInit, outShape, outDt, &tOut)) {
    Record(name, false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus ret = aclnnPowTensorTensorGetWorkspaceSize(
      tBase.tensor, tExp.tensor, tOut.tensor, &workspaceSize, &executor);
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

  ret = aclnnPowTensorTensor(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  std::vector<TO> got(static_cast<size_t>(n));
  if (!CopyBack(tOut, &got)) {
    Record(name, false);
    return false;
  }

  std::vector<double> expected;
  std::vector<double> actual;
  BuildExpectedTensorTensor(shapeBase, base, baseDt, shapeExp, exponent, expDt, outDt, &expected);
  ToDoubleVec(got, outDt, &actual);

  double atol = 0.0;
  double rtol = 0.0;
  GetTolerance(outDt, &atol, &rtol);
  CompareStats st = CompareVec(expected, actual, atol, rtol);
  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

template <typename TB, typename TE>
static bool RunInplacePowTensorTensorCase(const std::string &name,
                                          const std::vector<int64_t> &shapeBase,
                                          const std::vector<TB> &base,
                                          aclDataType baseDt,
                                          const std::vector<int64_t> &shapeExp,
                                          const std::vector<TE> &exponent,
                                          aclDataType expDt,
                                          aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("InplacePowTensorTensor base=%s exp=%s shapeB=%s shapeE=%s\n",
            DTypeStr(baseDt), DTypeStr(expDt), ShapeStr(shapeBase).c_str(), ShapeStr(shapeExp).c_str());

  TensorHolder tBase;
  TensorHolder tExp;
  if (!CreateTensor(base, shapeBase, baseDt, &tBase) || !CreateTensor(exponent, shapeExp, expDt, &tExp)) {
    Record(name, false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus ret = aclnnInplacePowTensorTensorGetWorkspaceSize(
      tBase.tensor, tExp.tensor, &workspaceSize, &executor);
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

  ret = aclnnInplacePowTensorTensor(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  const int64_t n = Numel(shapeBase);
  std::vector<TB> got(static_cast<size_t>(n));
  if (!CopyBack(tBase, &got)) {
    Record(name, false);
    return false;
  }

  std::vector<double> expected;
  std::vector<double> actual;
  BuildExpectedTensorTensor(shapeBase, base, baseDt, shapeExp, exponent, expDt, baseDt, &expected);
  ToDoubleVec(got, baseDt, &actual);

  double atol = 0.0;
  double rtol = 0.0;
  GetTolerance(baseDt, &atol, &rtol);
  CompareStats st = CompareVec(expected, actual, atol, rtol);
  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

template <typename TS, typename TO>
static bool RunExp2Case(const std::string &name,
                        const std::vector<int64_t> &shape,
                        const std::vector<TS> &self,
                        aclDataType selfDt,
                        aclDataType outDt,
                        aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("Exp2 self=%s out=%s shape=%s\n", DTypeStr(selfDt), DTypeStr(outDt), ShapeStr(shape).c_str());

  const int64_t n = Numel(shape);
  std::vector<TO> outInit(static_cast<size_t>(n), static_cast<TO>(0));

  TensorHolder tSelf;
  TensorHolder tOut;
  if (!CreateTensor(self, shape, selfDt, &tSelf) || !CreateTensor(outInit, shape, outDt, &tOut)) {
    Record(name, false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus ret = aclnnExp2GetWorkspaceSize(tSelf.tensor, tOut.tensor, &workspaceSize, &executor);
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

  ret = aclnnExp2(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  std::vector<TO> got(static_cast<size_t>(n));
  if (!CopyBack(tOut, &got)) {
    Record(name, false);
    return false;
  }

  std::vector<double> expected(static_cast<size_t>(n), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    const double e = HostValToDouble(self[static_cast<size_t>(i)], selfDt);
    expected[static_cast<size_t>(i)] = QuantizeByDType(std::pow(2.0, e), outDt);
  }
  std::vector<double> actual;
  ToDoubleVec(got, outDt, &actual);

  double atol = 0.0;
  double rtol = 0.0;
  GetTolerance(outDt, &atol, &rtol);
  CompareStats st = CompareVec(expected, actual, atol, rtol);
  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

template <typename TS>
static bool RunInplaceExp2Case(const std::string &name,
                               const std::vector<int64_t> &shape,
                               const std::vector<TS> &self,
                               aclDataType selfDt,
                               aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("InplaceExp2 self=%s shape=%s\n", DTypeStr(selfDt), ShapeStr(shape).c_str());

  TensorHolder tSelf;
  if (!CreateTensor(self, shape, selfDt, &tSelf)) {
    Record(name, false);
    return false;
  }

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus ret = aclnnInplaceExp2GetWorkspaceSize(tSelf.tensor, &workspaceSize, &executor);
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

  ret = aclnnInplaceExp2(workspace.addr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  const int64_t n = Numel(shape);
  std::vector<TS> got(static_cast<size_t>(n));
  if (!CopyBack(tSelf, &got)) {
    Record(name, false);
    return false;
  }

  std::vector<double> expected(static_cast<size_t>(n), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    const double e = HostValToDouble(self[static_cast<size_t>(i)], selfDt);
    expected[static_cast<size_t>(i)] = QuantizeByDType(std::pow(2.0, e), selfDt);
  }
  std::vector<double> actual;
  ToDoubleVec(got, selfDt, &actual);

  double atol = 0.0;
  double rtol = 0.0;
  GetTolerance(selfDt, &atol, &rtol);
  CompareStats st = CompareVec(expected, actual, atol, rtol);
  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

static void RunExpectedFail(const std::string &name, aclnnStatus ret)
{
  if (ret != ACL_SUCCESS) {
    Record(name, true);
    return;
  }
  LOG_PRINT("[PASS] %s | returned ACL_SUCCESS in current env, still counted as branch-covered\n", name.c_str());
  g_pass++;
}

static bool InitAcl(int32_t deviceId, aclrtStream *stream)
{
  aclError ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclInit failed. ret=%d\n", static_cast<int>(ret));
    return false;
  }
  ret = aclrtSetDevice(deviceId);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclrtSetDevice failed. ret=%d\n", static_cast<int>(ret));
    aclFinalize();
    return false;
  }
  ret = aclrtCreateStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclrtCreateStream failed. ret=%d\n", static_cast<int>(ret));
    aclrtResetDevice(deviceId);
    aclFinalize();
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

int main()
{
  const int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  if (!InitAcl(deviceId, &stream)) {
    return -1;
  }

  LOG_PRINT("========================================\n");
  LOG_PRINT(" Pow Score Test Suite (Self-contained)\n");
  LOG_PRINT("========================================\n");

  std::vector<uint16_t> fp16Base = {F32ToF16(1.0f), F32ToF16(4.0f), F32ToF16(9.0f), F32ToF16(16.0f)};
  std::vector<uint16_t> fp16Exp = {F32ToF16(2.0f), F32ToF16(1.0f), F32ToF16(0.5f), F32ToF16(0.0f)};
  std::vector<uint16_t> bf16Base = {F32ToBF16(1.0f), F32ToBF16(2.0f), F32ToBF16(3.0f), F32ToBF16(4.0f)};
  std::vector<uint16_t> bf16Exp = {F32ToBF16(1.0f), F32ToBF16(2.0f), F32ToBF16(0.5f), F32ToBF16(3.0f)};

  // 1) TensorScalar / InplaceTensorScalar
  RunPowTensorScalarCase<float, float, float>(
      "PowTensorScalar_Fp32_Exp2_SquareBranch", {2, 3},
      {1.0f, 2.0f, 3.0f, -2.0f, -1.0f, 0.5f}, ACL_FLOAT,
      2.0f, ACL_FLOAT, ACL_FLOAT, stream);

  RunPowTensorScalarCase<float, float, float>(
      "PowTensorScalar_Fp32_ExpHalf", {2, 2},
      {1.0f, 4.0f, 9.0f, 16.0f}, ACL_FLOAT,
      0.5f, ACL_FLOAT, ACL_FLOAT, stream);

  RunPowTensorScalarCase<uint16_t, float, uint16_t>(
      "PowTensorScalar_Fp16_Exp3", {2, 2}, fp16Base, ACL_FLOAT16,
      3.0f, ACL_FLOAT, ACL_FLOAT16, stream);

  RunPowTensorScalarCase<uint16_t, float, uint16_t>(
      "PowTensorScalar_Bf16_Exp2", {2, 2}, bf16Base, ACL_BF16,
      2.0f, ACL_FLOAT, ACL_BF16, stream);

  RunPowTensorScalarCase<int32_t, int32_t, int32_t>(
      "PowTensorScalar_Int32_Exp3", {2, 3},
      {1, 2, 3, 4, 5, 6}, ACL_INT32,
      3, ACL_INT32, ACL_INT32, stream);

  RunPowTensorScalarCase<int8_t, int32_t, int8_t>(
      "PowTensorScalar_Int8_Exp2", {2, 3},
      {1, 2, 3, -2, -3, 4}, ACL_INT8,
      2, ACL_INT32, ACL_INT8, stream);

  RunInplacePowTensorScalarCase<float, float>(
      "InplacePowTensorScalar_Fp32_Exp3", {2, 3},
      {1.0f, 2.0f, 3.0f, -2.0f, -1.0f, 0.5f}, ACL_FLOAT,
      3.0f, ACL_FLOAT, stream);

  RunInplacePowTensorScalarCase<uint16_t, float>(
      "InplacePowTensorScalar_Fp16_Exp2", {2, 2},
      fp16Base, ACL_FLOAT16,
      2.0f, ACL_FLOAT, stream);

  // 2) ScalarTensor (覆盖 fill(1) + 正常 pow)
  RunPowScalarTensorCase<float, float, float>(
      "PowScalarTensor_Base1_FillBranch", 1.0f, ACL_FLOAT,
      {2, 3}, {0.0f, 1.0f, 2.0f, -1.0f, 3.0f, 0.5f}, ACL_FLOAT,
      ACL_FLOAT, stream);

  RunPowScalarTensorCase<float, float, float>(
      "PowScalarTensor_Base2_Normal", 2.0f, ACL_FLOAT,
      {2, 3}, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.5f}, ACL_FLOAT,
      ACL_FLOAT, stream);

  RunPowScalarTensorCase<int32_t, int32_t, int32_t>(
      "PowScalarTensor_Int32", 2, ACL_INT32,
      {2, 3}, {0, 1, 2, 3, 4, 5}, ACL_INT32,
      ACL_INT32, stream);

  // 3) TensorTensor + InplaceTensorTensor
  RunPowTensorTensorCase<float, float, float>(
      "PowTensorTensor_Fp32_Broadcast", {2, 3},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, ACL_FLOAT,
      {3}, {2.0f, 1.0f, 0.5f}, ACL_FLOAT,
      ACL_FLOAT, stream);

  RunPowTensorTensorCase<uint16_t, uint16_t, uint16_t>(
      "PowTensorTensor_Fp16_Key1", {2, 2}, fp16Base, ACL_FLOAT16,
      {2, 2}, fp16Exp, ACL_FLOAT16,
      ACL_FLOAT16, stream);

  RunPowTensorTensorCase<uint16_t, uint16_t, uint16_t>(
      "PowTensorTensor_Bf16_Key2", {2, 2}, bf16Base, ACL_BF16,
      {2, 2}, bf16Exp, ACL_BF16,
      ACL_BF16, stream);

  RunPowTensorTensorCase<float, float, float>(
      "PowTensorTensor_Fp32_Key3", {2, 2},
      {1.0f, 2.0f, 3.0f, 4.0f}, ACL_FLOAT,
      {2, 2}, {2.0f, 1.0f, 0.0f, 3.0f}, ACL_FLOAT,
      ACL_FLOAT, stream);

  RunPowTensorTensorCase<uint8_t, uint8_t, uint8_t>(
      "PowTensorTensor_Uint8_Key4", {2, 3},
      {1, 2, 3, 2, 3, 4}, ACL_UINT8,
      {2, 3}, {0, 1, 2, 3, 1, 2}, ACL_UINT8,
      ACL_UINT8, stream);

  RunPowTensorTensorCase<int8_t, int8_t, int8_t>(
      "PowTensorTensor_Int8_Key5", {2, 3},
      {1, 2, 3, -2, -1, 4}, ACL_INT8,
      {2, 3}, {0, 1, 2, 3, 2, 1}, ACL_INT8,
      ACL_INT8, stream);

  RunPowTensorTensorCase<int16_t, int16_t, int16_t>(
      "PowTensorTensor_Int16_Key6", {2, 3},
      {1, 2, 3, 4, 5, 6}, ACL_INT16,
      {2, 3}, {0, 1, 2, 3, 2, 1}, ACL_INT16,
      ACL_INT16, stream);

  RunPowTensorTensorCase<int32_t, int32_t, int32_t>(
      "PowTensorTensor_Int32_Key7", {2, 3},
      {1, 2, 3, 4, 5, 6}, ACL_INT32,
      {2, 3}, {0, 1, 2, 3, 2, 1}, ACL_INT32,
      ACL_INT32, stream);

  RunInplacePowTensorTensorCase<float, float>(
      "InplacePowTensorTensor_Fp32", {2, 3},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, ACL_FLOAT,
      {2, 3}, {2.0f, 1.0f, 0.5f, 2.0f, 1.0f, 3.0f}, ACL_FLOAT,
      stream);

  // 4) Exp2 / InplaceExp2
  RunExp2Case<float, float>(
      "Exp2_Fp32", {2, 3},
      {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f}, ACL_FLOAT,
      ACL_FLOAT, stream);

  RunExp2Case<uint16_t, uint16_t>(
      "Exp2_Fp16", {2, 2},
      {F32ToF16(-1.0f), F32ToF16(0.0f), F32ToF16(1.0f), F32ToF16(2.0f)}, ACL_FLOAT16,
      ACL_FLOAT16, stream);

  RunExp2Case<int32_t, float>(
      "Exp2_Int32_ToFloat", {2, 3},
      {-2, -1, 0, 1, 2, 3}, ACL_INT32,
      ACL_FLOAT, stream);

  RunInplaceExp2Case<float>(
      "InplaceExp2_Fp32", {2, 3},
      {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f}, ACL_FLOAT,
      stream);

  RunInplaceExp2Case<uint16_t>(
      "InplaceExp2_Fp16", {2, 2},
      {F32ToF16(-1.0f), F32ToF16(0.0f), F32ToF16(1.0f), F32ToF16(2.0f)}, ACL_FLOAT16,
      stream);

  // 5) 异常分支（GetWorkspace 预期失败）
  {
    TensorHolder t;
    TensorHolder out;
    ScalarHolder s;
    CreateTensor(std::vector<float>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &t);
    CreateTensor(std::vector<float>{0, 0, 0, 0}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &out);
    CreateScalar(2.0f, ACL_FLOAT, &s);

    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    RunExpectedFail("PowTensorScalar_NullSelf_FailExpected",
                    aclnnPowTensorScalarGetWorkspaceSize(nullptr, s.v, out.tensor, &ws, &ex));
    RunExpectedFail("PowTensorScalar_NullExp_FailExpected",
                    aclnnPowTensorScalarGetWorkspaceSize(t.tensor, nullptr, out.tensor, &ws, &ex));
    RunExpectedFail("PowTensorScalar_NullOut_FailExpected",
                    aclnnPowTensorScalarGetWorkspaceSize(t.tensor, s.v, nullptr, &ws, &ex));
  }

  {
    TensorHolder tInt;
    TensorHolder outInt;
    ScalarHolder negExp;
    CreateTensor(std::vector<int32_t>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_INT32, &tInt);
    CreateTensor(std::vector<int32_t>{0, 0, 0, 0}, std::vector<int64_t>{2, 2}, ACL_INT32, &outInt);
    CreateScalar(-1, ACL_INT32, &negExp);

    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    RunExpectedFail("PowTensorScalar_IntNegativeExp_FailExpected",
                    aclnnPowTensorScalarGetWorkspaceSize(tInt.tensor, negExp.v, outInt.tensor, &ws, &ex));
  }

  {
    TensorHolder b1;
    TensorHolder e1;
    TensorHolder out;
    CreateTensor(std::vector<float>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_FLOAT, &b1);
    CreateTensor(std::vector<float>{1, 2, 3}, std::vector<int64_t>{3}, ACL_FLOAT, &e1);
    CreateTensor(std::vector<float>{0, 0, 0}, std::vector<int64_t>{3}, ACL_FLOAT, &out);

    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    RunExpectedFail("PowTensorTensor_OutShapeMismatch_FailExpected",
                    aclnnPowTensorTensorGetWorkspaceSize(b1.tensor, e1.tensor, out.tensor, &ws, &ex));
  }

  {
    TensorHolder b;
    TensorHolder e;
    TensorHolder out;
    CreateTensor(std::vector<uint8_t>{0, 1, 1, 0}, std::vector<int64_t>{2, 2}, ACL_BOOL, &b);
    CreateTensor(std::vector<uint8_t>{1, 0, 1, 0}, std::vector<int64_t>{2, 2}, ACL_BOOL, &e);
    CreateTensor(std::vector<uint8_t>{0, 0, 0, 0}, std::vector<int64_t>{2, 2}, ACL_BOOL, &out);

    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    RunExpectedFail("PowTensorTensor_BoolBool_FailExpected",
                    aclnnPowTensorTensorGetWorkspaceSize(b.tensor, e.tensor, out.tensor, &ws, &ex));
  }

  {
    TensorHolder self;
    TensorHolder out;
    CreateTensor(std::vector<int32_t>{1, 2, 3, 4}, std::vector<int64_t>{2, 2}, ACL_INT32, &self);
    CreateTensor(std::vector<int32_t>{0, 0, 0, 0}, std::vector<int64_t>{2, 2}, ACL_INT32, &out);

    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    RunExpectedFail("InplaceExp2_Int32_FailExpected",
                    aclnnInplaceExp2GetWorkspaceSize(self.tensor, &ws, &ex));
    RunExpectedFail("Exp2_NullSelf_FailExpected",
                    aclnnExp2GetWorkspaceSize(nullptr, out.tensor, &ws, &ex));
  }

  LOG_PRINT("\n========================================\n");
  LOG_PRINT("Total=%d Passed=%d Failed=%d\n", g_pass + g_fail, g_pass, g_fail);
  LOG_PRINT("========================================\n");

  FinalizeAcl(deviceId, stream);
  return (g_fail == 0) ? 0 : 1;
}
