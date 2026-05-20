/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

#define LOG_PRINT(fmt, ...)     \
  do {                          \
    printf(fmt, ##__VA_ARGS__); \
  } while (0)

static int g_pass = 0;
static int g_fail = 0;

struct CompareStats {
  bool pass = true;
  double maxAbs = 0.0;
  double maxRel = 0.0;
  int64_t bad = -1;
};

struct TensorHolder {
  void *addr = nullptr;
  aclTensor *tensor = nullptr;
  TensorHolder() = default;
  TensorHolder(const TensorHolder &) = delete;
  TensorHolder &operator=(const TensorHolder &) = delete;
  ~TensorHolder()
  {
    if (tensor != nullptr) aclDestroyTensor(tensor);
    if (addr != nullptr) aclrtFree(addr);
  }
};

struct WsHolder {
  void *addr = nullptr;
  WsHolder() = default;
  WsHolder(const WsHolder &) = delete;
  WsHolder &operator=(const WsHolder &) = delete;
  ~WsHolder()
  {
    if (addr != nullptr) aclrtFree(addr);
  }
};

struct ScalarHolder {
  aclScalar *v = nullptr;
  ScalarHolder() = default;
  ScalarHolder(const ScalarHolder &) = delete;
  ScalarHolder &operator=(const ScalarHolder &) = delete;
  ~ScalarHolder()
  {
    if (v != nullptr) aclDestroyScalar(v);
  }
};

std::string ShapeStr(const std::vector<int64_t> &s)
{
  if (s.empty()) return "[]";
  std::string r = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    r += std::to_string(s[i]);
    if (i + 1 != s.size()) r += ", ";
  }
  r += "]";
  return r;
}

const char *DTypeStr(aclDataType t)
{
  switch (t) {
    case ACL_FLOAT16:
      return "FLOAT16";
    case ACL_BF16:
      return "BF16";
    case ACL_FLOAT:
      return "FLOAT32";
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

int64_t Numel(const std::vector<int64_t> &shape)
{
  if (shape.empty()) return 1;
  int64_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) return -1;
    n *= shape[i];
  }
  return n;
}

std::vector<int64_t> Strides(const std::vector<int64_t> &shape)
{
  std::vector<int64_t> s(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) s[static_cast<size_t>(i)] = s[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  return s;
}

bool BroadcastShape(const std::vector<int64_t> &a, const std::vector<int64_t> &b, std::vector<int64_t> *o)
{
  if (o == nullptr) return false;
  o->clear();
  size_t ra = a.size(), rb = b.size(), ro = (ra > rb ? ra : rb);
  o->resize(ro, 1);
  for (size_t i = 0; i < ro; ++i) {
    int64_t da = (i < ro - ra) ? 1 : a[i - (ro - ra)];
    int64_t db = (i < ro - rb) ? 1 : b[i - (ro - rb)];
    if (da != db && da != 1 && db != 1) return false;
    (*o)[i] = da > db ? da : db;
  }
  return true;
}

void Unravel(int64_t x, const std::vector<int64_t> &shape, std::vector<int64_t> *idx)
{
  idx->assign(shape.size(), 0);
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    int64_t d = shape[static_cast<size_t>(i)];
    if (d > 0) {
      (*idx)[static_cast<size_t>(i)] = x % d;
      x /= d;
    }
  }
}

int64_t BcOffset(const std::vector<int64_t> &outIdx, const std::vector<int64_t> &inShape, const std::vector<int64_t> &inStride)
{
  if (inShape.empty()) return 0;
  int64_t off = 0;
  size_t ro = outIdx.size(), ri = inShape.size();
  for (size_t i = 0; i < ri; ++i) {
    size_t op = ro - ri + i;
    int64_t id = (inShape[i] == 1) ? 0 : outIdx[op];
    off += id * inStride[i];
  }
  return off;
}

float F16ToF32(uint16_t h)
{
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
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

uint16_t F32ToF16(float f)
{
  uint32_t x = 0;
  std::memcpy(&x, &f, sizeof(float));
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t halfMant = mant >> shift;
    if ((mant >> (shift - 1)) & 1u) halfMant++;
    return static_cast<uint16_t>(sign | halfMant);
  }
  if (exp >= 31) {
    bool nan = ((x & 0x7FFFFFFFu) > 0x7F800000u);
    return static_cast<uint16_t>(nan ? (sign | 0x7E00u) : (sign | 0x7C00u));
  }
  mant += 0x1000u;
  if (mant & 0x800000u) {
    mant = 0;
    exp++;
  }
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float BF16ToF32(uint16_t b)
{
  uint32_t x = static_cast<uint32_t>(b) << 16;
  float f = 0.0f;
  std::memcpy(&f, &x, sizeof(float));
  return f;
}

uint16_t F32ToBF16(float f)
{
  uint32_t x = 0;
  std::memcpy(&x, &f, sizeof(float));
  uint32_t lsb = (x >> 16) & 1u;
  x += 0x7FFFu + lsb;
  return static_cast<uint16_t>(x >> 16);
}

template <typename T>
double ToDouble(T v, aclDataType dt)
{
  if (dt == ACL_FLOAT16) return static_cast<double>(F16ToF32(static_cast<uint16_t>(v)));
  if (dt == ACL_BF16) return static_cast<double>(BF16ToF32(static_cast<uint16_t>(v)));
  return static_cast<double>(v);
}

void Tolerance(aclDataType dt, double *atol, double *rtol)
{
  if (dt == ACL_FLOAT16 || dt == ACL_BF16) {
    *atol = 1e-3;
    *rtol = 1e-3;
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

CompareStats CompareVec(const std::vector<double> &exp, const std::vector<double> &got, double atol, double rtol)
{
  CompareStats s;
  if (exp.size() != got.size()) {
    s.pass = false;
    return s;
  }
  for (size_t i = 0; i < exp.size(); ++i) {
    double e = exp[i], g = got[i];
    bool en = std::isnan(e), gn = std::isnan(g);
    if (en || gn) {
      if (!(en && gn)) {
        s.pass = false;
        s.bad = static_cast<int64_t>(i);
        return s;
      }
      continue;
    }
    bool ei = std::isinf(e), gi = std::isinf(g);
    if (ei || gi) {
      bool ok = ei && gi && (std::signbit(e) == std::signbit(g));
      if (!ok) {
        s.pass = false;
        s.bad = static_cast<int64_t>(i);
        return s;
      }
      continue;
    }
    double ad = std::fabs(g - e);
    double rd = ad / std::fmax(1.0, std::fabs(e));
    if (ad > s.maxAbs) s.maxAbs = ad;
    if (rd > s.maxRel) s.maxRel = rd;
    if (!(ad <= atol || rd <= rtol)) {
      s.pass = false;
      s.bad = static_cast<int64_t>(i);
      return s;
    }
  }
  return s;
}

void Record(const std::string &name, bool ok, double maxAbs = 0.0, double maxRel = 0.0)
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
bool CreateTensor(const std::vector<T> &host, const std::vector<int64_t> &shape, aclDataType dt, TensorHolder *t)
{
  if (t == nullptr) return false;
  int64_t n = Numel(shape);
  if (n < 0 || n != static_cast<int64_t>(host.size())) {
    LOG_PRINT("[ERROR] shape/data mismatch shape=%s numel=%lld data=%zu\n", ShapeStr(shape).c_str(), static_cast<long long>(n), host.size());
    return false;
  }
  size_t bytes = host.size() * sizeof(T);
  if (bytes > 0) {
    aclError r = aclrtMalloc(&(t->addr), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (r != ACL_SUCCESS) return false;
    r = aclrtMemcpy(t->addr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (r != ACL_SUCCESS) return false;
  }
  std::vector<int64_t> st = Strides(shape);
  const int64_t *sp = shape.empty() ? nullptr : shape.data();
  const int64_t *tp = st.empty() ? nullptr : st.data();
  t->tensor = aclCreateTensor(sp, shape.size(), dt, tp, 0, aclFormat::ACL_FORMAT_ND, sp, shape.size(), t->addr);
  return t->tensor != nullptr;
}

template <typename T>
bool CopyBack(const TensorHolder &t, std::vector<T> *host)
{
  if (host == nullptr) return false;
  size_t bytes = host->size() * sizeof(T);
  if (bytes == 0) return true;
  aclError r = aclrtMemcpy(host->data(), bytes, t.addr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
  return r == ACL_SUCCESS;
}

template <typename T>
void ToDoubleVec(const std::vector<T> &src, aclDataType dt, std::vector<double> *dst)
{
  dst->resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) (*dst)[i] = ToDouble(src[i], dt);
}

template <typename T>
void BuildMulExpected(const std::vector<int64_t> &sa, const std::vector<T> &a, const std::vector<int64_t> &sb,
                      const std::vector<T> &b, aclDataType dt, std::vector<double> *exp)
{
  std::vector<int64_t> so;
  BroadcastShape(sa, sb, &so);
  int64_t no = Numel(so);
  exp->assign(static_cast<size_t>(no), 0.0);
  std::vector<int64_t> ia;
  std::vector<int64_t> stA = Strides(sa), stB = Strides(sb);
  for (int64_t i = 0; i < no; ++i) {
    Unravel(i, so, &ia);
    int64_t oa = BcOffset(ia, sa, stA), ob = BcOffset(ia, sb, stB);
    (*exp)[static_cast<size_t>(i)] = ToDouble(a[static_cast<size_t>(oa)], dt) * ToDouble(b[static_cast<size_t>(ob)], dt);
  }
}

template <typename TA, typename TB>
void BuildMulExpectedMixed(const std::vector<int64_t> &sa, const std::vector<TA> &a, aclDataType dta,
                           const std::vector<int64_t> &sb, const std::vector<TB> &b, aclDataType dtb,
                           std::vector<double> *exp)
{
  std::vector<int64_t> so;
  BroadcastShape(sa, sb, &so);
  int64_t no = Numel(so);
  exp->assign(static_cast<size_t>(no), 0.0);
  std::vector<int64_t> io;
  std::vector<int64_t> stA = Strides(sa), stB = Strides(sb);
  for (int64_t i = 0; i < no; ++i) {
    Unravel(i, so, &io);
    int64_t oa = BcOffset(io, sa, stA);
    int64_t ob = BcOffset(io, sb, stB);
    (*exp)[static_cast<size_t>(i)] = ToDouble(a[static_cast<size_t>(oa)], dta) * ToDouble(b[static_cast<size_t>(ob)], dtb);
  }
}

template <typename T>
bool RunMul(const std::string &name, const std::vector<int64_t> &sa, const std::vector<T> &a, const std::vector<int64_t> &sb,
            const std::vector<T> &b, aclDataType dt, aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("op=Mul dtype=%s sa=%s sb=%s\n", DTypeStr(dt), ShapeStr(sa).c_str(), ShapeStr(sb).c_str());

  std::vector<int64_t> so;
  if (!BroadcastShape(sa, sb, &so)) {
    Record(name, false);
    return false;
  }
  int64_t no = Numel(so);
  std::vector<T> outInit(static_cast<size_t>(no), static_cast<T>(0));

  TensorHolder ta, tb, to;
  if (!CreateTensor(a, sa, dt, &ta) || !CreateTensor(b, sb, dt, &tb) || !CreateTensor(outInit, so, dt, &to)) {
    Record(name, false);
    return false;
  }
  uint64_t ws = 0;
  aclOpExecutor *ex = nullptr;
  aclError r = aclnnMulGetWorkspaceSize(ta.tensor, tb.tensor, to.tensor, &ws, &ex);
  if (r != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }
  WsHolder w;
  if (ws > 0) {
    r = aclrtMalloc(&(w.addr), ws, ACL_MEM_MALLOC_HUGE_FIRST);
    if (r != ACL_SUCCESS) {
      Record(name, false);
      return false;
    }
  }
  r = aclnnMul(w.addr, ws, ex, stream);
  if (r != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }
  std::vector<T> got(static_cast<size_t>(no));
  if (!CopyBack(to, &got)) {
    Record(name, false);
    return false;
  }
  std::vector<double> expD, gotD;
  BuildMulExpected(sa, a, sb, b, dt, &expD);
  ToDoubleVec(got, dt, &gotD);
  double atol = 0.0, rtol = 0.0;
  Tolerance(dt, &atol, &rtol);
  CompareStats st = CompareVec(expD, gotD, atol, rtol);
  if (!st.pass && st.bad >= 0) LOG_PRINT("mismatch idx=%lld exp=%.9g got=%.9g\n", static_cast<long long>(st.bad), expD[static_cast<size_t>(st.bad)], gotD[static_cast<size_t>(st.bad)]);
  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

template <typename TA, typename TB, typename TO>
bool RunMulMixed(const std::string &name, const std::vector<int64_t> &sa, const std::vector<TA> &a, aclDataType dta,
                 const std::vector<int64_t> &sb, const std::vector<TB> &b, aclDataType dtb, aclDataType dto,
                 aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("op=Mul dta=%s dtb=%s dto=%s sa=%s sb=%s\n",
            DTypeStr(dta), DTypeStr(dtb), DTypeStr(dto), ShapeStr(sa).c_str(), ShapeStr(sb).c_str());

  std::vector<int64_t> so;
  if (!BroadcastShape(sa, sb, &so)) {
    Record(name, false);
    return false;
  }

  int64_t no = Numel(so);
  std::vector<TO> outInit(static_cast<size_t>(no), static_cast<TO>(0));
  TensorHolder ta, tb, to;
  if (!CreateTensor(a, sa, dta, &ta) || !CreateTensor(b, sb, dtb, &tb) || !CreateTensor(outInit, so, dto, &to)) {
    Record(name, false);
    return false;
  }

  uint64_t ws = 0;
  aclOpExecutor *ex = nullptr;
  aclError r = aclnnMulGetWorkspaceSize(ta.tensor, tb.tensor, to.tensor, &ws, &ex);
  if (r != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }
  WsHolder w;
  if (ws > 0 && aclrtMalloc(&(w.addr), ws, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }
  r = aclnnMul(w.addr, ws, ex, stream);
  if (r != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }

  std::vector<TO> got(static_cast<size_t>(no));
  if (!CopyBack(to, &got)) {
    Record(name, false);
    return false;
  }

  std::vector<double> expD, gotD;
  BuildMulExpectedMixed(sa, a, dta, sb, b, dtb, &expD);
  ToDoubleVec(got, dto, &gotD);
  double atol = 0.0, rtol = 0.0;
  Tolerance(dto, &atol, &rtol);
  CompareStats st = CompareVec(expD, gotD, atol, rtol);
  if (!st.pass && st.bad >= 0) {
    LOG_PRINT("mismatch idx=%lld exp=%.9g got=%.9g\n",
              static_cast<long long>(st.bad), expD[static_cast<size_t>(st.bad)], gotD[static_cast<size_t>(st.bad)]);
  }
  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

template <typename TA, typename TB, typename TO>
bool RunMulStatusOnly(const std::string &name, const std::vector<int64_t> &sa, const std::vector<TA> &a, aclDataType dta,
                      const std::vector<int64_t> &sb, const std::vector<TB> &b, aclDataType dtb,
                      const std::vector<int64_t> &so, const std::vector<TO> &outInit, aclDataType dto,
                      aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  LOG_PRINT("op=Mul(StatusOnly) dta=%s dtb=%s dto=%s\n", DTypeStr(dta), DTypeStr(dtb), DTypeStr(dto));
  TensorHolder ta, tb, to;
  if (!CreateTensor(a, sa, dta, &ta) || !CreateTensor(b, sb, dtb, &tb) || !CreateTensor(outInit, so, dto, &to)) {
    Record(name, false);
    return false;
  }
  uint64_t ws = 0;
  aclOpExecutor *ex = nullptr;
  aclError r = aclnnMulGetWorkspaceSize(ta.tensor, tb.tensor, to.tensor, &ws, &ex);
  if (r != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }
  WsHolder w;
  if (ws > 0 && aclrtMalloc(&(w.addr), ws, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }
  r = aclnnMul(w.addr, ws, ex, stream);
  bool ok = (r == ACL_SUCCESS && aclrtSynchronizeStream(stream) == ACL_SUCCESS);
  Record(name, ok);
  return ok;
}

template <typename T>
bool RunMuls(const std::string &name, const std::vector<int64_t> &s, const std::vector<T> &a, float scalar, aclDataType dt,
             aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  TensorHolder ta, to;
  int64_t n = Numel(s);
  std::vector<T> out(static_cast<size_t>(n), static_cast<T>(0));
  if (!CreateTensor(a, s, dt, &ta) || !CreateTensor(out, s, dt, &to)) {
    Record(name, false);
    return false;
  }
  ScalarHolder sc;
  sc.v = aclCreateScalar(&scalar, ACL_FLOAT);
  if (sc.v == nullptr) {
    Record(name, false);
    return false;
  }
  uint64_t ws = 0;
  aclOpExecutor *ex = nullptr;
  aclError r = aclnnMulsGetWorkspaceSize(ta.tensor, sc.v, to.tensor, &ws, &ex);
  if (r != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }
  WsHolder w;
  if (ws > 0 && aclrtMalloc(&(w.addr), ws, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }
  r = aclnnMuls(w.addr, ws, ex, stream);
  if (r != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }
  std::vector<T> got(static_cast<size_t>(n));
  if (!CopyBack(to, &got)) {
    Record(name, false);
    return false;
  }
  std::vector<double> exp(static_cast<size_t>(n)), gotD;
  for (int64_t i = 0; i < n; ++i) exp[static_cast<size_t>(i)] = ToDouble(a[static_cast<size_t>(i)], dt) * scalar;
  ToDoubleVec(got, dt, &gotD);
  double atol = 0.0, rtol = 0.0;
  Tolerance(dt, &atol, &rtol);
  CompareStats st = CompareVec(exp, gotD, atol, rtol);
  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

template <typename T>
bool RunInplaceMul(const std::string &name, const std::vector<int64_t> &sa, const std::vector<T> &a, const std::vector<int64_t> &sb,
                   const std::vector<T> &b, aclDataType dt, aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  std::vector<int64_t> so;
  if (!BroadcastShape(sa, sb, &so) || so != sa) {
    Record(name, false);
    return false;
  }
  TensorHolder ta, tb;
  if (!CreateTensor(a, sa, dt, &ta) || !CreateTensor(b, sb, dt, &tb)) {
    Record(name, false);
    return false;
  }
  uint64_t ws = 0;
  aclOpExecutor *ex = nullptr;
  aclError r = aclnnInplaceMulGetWorkspaceSize(ta.tensor, tb.tensor, &ws, &ex);
  if (r != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }
  WsHolder w;
  if (ws > 0 && aclrtMalloc(&(w.addr), ws, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }
  r = aclnnInplaceMul(w.addr, ws, ex, stream);
  if (r != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }
  int64_t n = Numel(sa);
  std::vector<T> got(static_cast<size_t>(n));
  if (!CopyBack(ta, &got)) {
    Record(name, false);
    return false;
  }
  std::vector<double> exp, gotD;
  BuildMulExpected(sa, a, sb, b, dt, &exp);
  ToDoubleVec(got, dt, &gotD);
  double atol = 0.0, rtol = 0.0;
  Tolerance(dt, &atol, &rtol);
  CompareStats st = CompareVec(exp, gotD, atol, rtol);
  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

template <typename T>
bool RunInplaceMuls(const std::string &name, const std::vector<int64_t> &s, const std::vector<T> &a, float scalar, aclDataType dt,
                    aclrtStream stream)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  TensorHolder ta;
  if (!CreateTensor(a, s, dt, &ta)) {
    Record(name, false);
    return false;
  }
  ScalarHolder sc;
  sc.v = aclCreateScalar(&scalar, ACL_FLOAT);
  if (sc.v == nullptr) {
    Record(name, false);
    return false;
  }
  uint64_t ws = 0;
  aclOpExecutor *ex = nullptr;
  aclError r = aclnnInplaceMulsGetWorkspaceSize(ta.tensor, sc.v, &ws, &ex);
  if (r != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }
  WsHolder w;
  if (ws > 0 && aclrtMalloc(&(w.addr), ws, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }
  r = aclnnInplaceMuls(w.addr, ws, ex, stream);
  if (r != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    Record(name, false);
    return false;
  }
  int64_t n = Numel(s);
  std::vector<T> got(static_cast<size_t>(n));
  if (!CopyBack(ta, &got)) {
    Record(name, false);
    return false;
  }
  std::vector<double> exp(static_cast<size_t>(n)), gotD;
  for (int64_t i = 0; i < n; ++i) exp[static_cast<size_t>(i)] = ToDouble(a[static_cast<size_t>(i)], dt) * scalar;
  ToDoubleVec(got, dt, &gotD);
  double atol = 0.0, rtol = 0.0;
  Tolerance(dt, &atol, &rtol);
  CompareStats st = CompareVec(exp, gotD, atol, rtol);
  Record(name, st.pass, st.maxAbs, st.maxRel);
  return st.pass;
}

template <typename T>
bool RunMulsExpectFail(const std::string &name, const std::vector<int64_t> &s, const std::vector<T> &a, float scalar,
                       const std::vector<int64_t> &outShape, aclDataType dt, aclrtStream stream, bool nullSelf = false)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  TensorHolder ta, to;
  if (!nullSelf && !CreateTensor(a, s, dt, &ta)) {
    Record(name, false);
    return false;
  }
  int64_t no = Numel(outShape);
  std::vector<T> out(static_cast<size_t>(no > 0 ? no : 0), static_cast<T>(0));
  if (!CreateTensor(out, outShape, dt, &to)) {
    Record(name, false);
    return false;
  }
  ScalarHolder sc;
  sc.v = aclCreateScalar(&scalar, ACL_FLOAT);
  if (sc.v == nullptr) {
    Record(name, false);
    return false;
  }

  uint64_t ws = 0;
  aclOpExecutor *ex = nullptr;
  aclTensor *selfPtr = nullSelf ? nullptr : ta.tensor;
  aclError r = aclnnMulsGetWorkspaceSize(selfPtr, sc.v, to.tensor, &ws, &ex);
  bool ok = false;
  if (r != ACL_SUCCESS) {
    ok = true;
  } else {
    WsHolder w;
    if (ws > 0 && aclrtMalloc(&(w.addr), ws, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      Record(name, false);
      return false;
    }
    r = aclnnMuls(w.addr, ws, ex, stream);
    ok = (r != ACL_SUCCESS) || (aclrtSynchronizeStream(stream) != ACL_SUCCESS);
  }
  Record(name, ok);
  return ok;
}

template <typename T>
bool RunInplaceMulExpectFail(const std::string &name, const std::vector<int64_t> &sa, const std::vector<T> &a,
                             const std::vector<int64_t> &sb, const std::vector<T> &b, aclDataType dt,
                             aclrtStream stream, bool nullSelf = false)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  TensorHolder ta, tb;
  if (!nullSelf && !CreateTensor(a, sa, dt, &ta)) {
    Record(name, false);
    return false;
  }
  if (!CreateTensor(b, sb, dt, &tb)) {
    Record(name, false);
    return false;
  }
  uint64_t ws = 0;
  aclOpExecutor *ex = nullptr;
  aclTensor *selfPtr = nullSelf ? nullptr : ta.tensor;
  aclError r = aclnnInplaceMulGetWorkspaceSize(selfPtr, tb.tensor, &ws, &ex);
  bool ok = false;
  if (r != ACL_SUCCESS) {
    ok = true;
  } else {
    WsHolder w;
    if (ws > 0 && aclrtMalloc(&(w.addr), ws, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      Record(name, false);
      return false;
    }
    r = aclnnInplaceMul(w.addr, ws, ex, stream);
    ok = (r != ACL_SUCCESS) || (aclrtSynchronizeStream(stream) != ACL_SUCCESS);
  }
  Record(name, ok);
  return ok;
}

template <typename T>
bool RunInplaceMulsExpectFail(const std::string &name, const std::vector<int64_t> &s, const std::vector<T> &a,
                              float scalar, aclDataType dt, aclrtStream stream, bool nullSelf = false)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  TensorHolder ta;
  if (!nullSelf && !CreateTensor(a, s, dt, &ta)) {
    Record(name, false);
    return false;
  }
  ScalarHolder sc;
  sc.v = aclCreateScalar(&scalar, ACL_FLOAT);
  if (sc.v == nullptr) {
    Record(name, false);
    return false;
  }

  uint64_t ws = 0;
  aclOpExecutor *ex = nullptr;
  aclTensor *selfPtr = nullSelf ? nullptr : ta.tensor;
  aclError r = aclnnInplaceMulsGetWorkspaceSize(selfPtr, sc.v, &ws, &ex);
  bool ok = false;
  if (r != ACL_SUCCESS) {
    ok = true;
  } else {
    WsHolder w;
    if (ws > 0 && aclrtMalloc(&(w.addr), ws, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      Record(name, false);
      return false;
    }
    r = aclnnInplaceMuls(w.addr, ws, ex, stream);
    ok = (r != ACL_SUCCESS) || (aclrtSynchronizeStream(stream) != ACL_SUCCESS);
  }
  Record(name, ok);
  return ok;
}

template <typename T>
bool RunMulExpectFail(const std::string &name, const std::vector<int64_t> &sa, const std::vector<T> &a, const std::vector<int64_t> &sb,
                      const std::vector<T> &b, const std::vector<int64_t> &so, aclDataType dt, aclrtStream stream,
                      bool nullSelf = false)
{
  LOG_PRINT("\n-- %s --\n", name.c_str());
  TensorHolder ta, tb, to;
  if (!nullSelf && !CreateTensor(a, sa, dt, &ta)) {
    Record(name, false);
    return false;
  }
  if (!CreateTensor(b, sb, dt, &tb)) {
    Record(name, false);
    return false;
  }
  int64_t no = Numel(so);
  std::vector<T> out(static_cast<size_t>(no > 0 ? no : 0), static_cast<T>(0));
  if (!CreateTensor(out, so, dt, &to)) {
    Record(name, false);
    return false;
  }
  uint64_t ws = 0;
  aclOpExecutor *ex = nullptr;
  aclTensor *selfPtr = nullSelf ? nullptr : ta.tensor;
  aclError r = aclnnMulGetWorkspaceSize(selfPtr, tb.tensor, to.tensor, &ws, &ex);
  bool ok = false;
  if (r != ACL_SUCCESS) {
    ok = true;
  } else {
    WsHolder w;
    if (ws > 0 && aclrtMalloc(&(w.addr), ws, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
      Record(name, false);
      return false;
    }
    r = aclnnMul(w.addr, ws, ex, stream);
    if (r != ACL_SUCCESS) {
      ok = true;
    } else {
      ok = (aclrtSynchronizeStream(stream) != ACL_SUCCESS);
    }
  }
  Record(name, ok);
  return ok;
}

template <typename T>
std::vector<T> Repeat(const std::vector<T> &p, int64_t n)
{
  std::vector<T> o;
  if (p.empty() || n <= 0) return o;
  o.resize(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) o[static_cast<size_t>(i)] = p[static_cast<size_t>(i % static_cast<int64_t>(p.size()))];
  return o;
}

bool Init(int32_t deviceId, aclrtStream *stream)
{
  aclError r = aclInit(nullptr);
  if (r != ACL_SUCCESS) return false;
  r = aclrtSetDevice(deviceId);
  if (r != ACL_SUCCESS) return false;
  r = aclrtCreateStream(stream);
  return r == ACL_SUCCESS;
}

void Done(int32_t deviceId, aclrtStream stream)
{
  if (stream != nullptr) aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  if (!Init(deviceId, &stream)) return -1;

  LOG_PRINT("========================================\n");
  LOG_PRINT(" Mul Score Boost Test Suite\n");
  LOG_PRINT("========================================\n");

  RunMul<float>("TestMul_Fp32_Basic_Success", {2, 3}, {1, 2, 3, 4, 5, 6}, {2, 3}, {2, 2, 2, 3, 3, 3}, ACL_FLOAT, stream);
  RunMul<float>("TestMul_Fp32_BroadcastRow_Success", {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3}, {2, 3, 4},
                ACL_FLOAT, stream);
  RunMul<float>("TestMul_Fp32_BroadcastCross_Success", {3, 1}, {1, 2, 3}, {1, 4}, {1, 2, 3, 4}, ACL_FLOAT, stream);
  RunMul<float>("TestMul_Fp32_Unaligned_13x27_Success", {13, 27}, Repeat<float>({-3.0f, -1.0f, 0.5f, 2.0f}, 13 * 27), {1, 27},
                Repeat<float>({0.5f, 1.0f, -2.0f}, 27), ACL_FLOAT, stream);
  RunMul<float>("TestMul_Fp32_Dim5_ContiguousFallback_Success", {1, 1, 1, 2, 3}, {1, 2, 3, 4, 5, 6},
                {1, 1, 1, 2, 3}, {2, 2, 2, 3, 3, 3}, ACL_FLOAT, stream);
  RunMul<float>("TestMul_Fp32_EmptyTensor_Success", {0, 4}, {}, {0, 4}, {}, ACL_FLOAT, stream);

  std::vector<float> la(1024LL * 1024LL), lb(1024LL * 1024LL);
  for (int64_t i = 0; i < 1024LL * 1024LL; ++i) {
    la[static_cast<size_t>(i)] = static_cast<float>((i % 97) - 48) * 0.25f;
    lb[static_cast<size_t>(i)] = static_cast<float>((i % 29) + 1) * 0.1f;
  }
  RunMul<float>("TestMul_Fp32_Large1024x1024_Success", {1024, 1024}, la, {1024, 1024}, lb, ACL_FLOAT, stream);

  RunMul<int32_t>("TestMul_Int32_Basic_Success", {2, 3}, {1, 2, 3, 4, 5, 6}, {2, 3}, {2, 3, 4, 5, 6, 7}, ACL_INT32, stream);
  RunMul<int64_t>("TestMul_Int64_Basic_Success", {2, 2}, {10, 20, 30, 40}, {2, 2}, {2, 3, 4, 5}, ACL_INT64, stream);
  RunMul<int16_t>("TestMul_Int16_Basic_Success", {2, 3}, {1, -2, 3, -4, 5, -6}, {2, 3}, {-1, 2, -3, 4, -5, 6}, ACL_INT16, stream);
  RunMul<int8_t>("TestMul_Int8_Basic_Success", {2, 3},
                 {static_cast<int8_t>(-3), static_cast<int8_t>(-2), static_cast<int8_t>(-1),
                  static_cast<int8_t>(1), static_cast<int8_t>(2), static_cast<int8_t>(3)},
                 {2, 3},
                 {static_cast<int8_t>(3), static_cast<int8_t>(2), static_cast<int8_t>(1),
                  static_cast<int8_t>(-1), static_cast<int8_t>(-2), static_cast<int8_t>(-3)},
                 ACL_INT8, stream);
  RunMul<uint8_t>("TestMul_Uint8_Basic_Success", {2, 3},
                  {static_cast<uint8_t>(1), static_cast<uint8_t>(2), static_cast<uint8_t>(3),
                   static_cast<uint8_t>(4), static_cast<uint8_t>(5), static_cast<uint8_t>(6)},
                  {2, 3},
                  {static_cast<uint8_t>(2), static_cast<uint8_t>(3), static_cast<uint8_t>(4),
                   static_cast<uint8_t>(5), static_cast<uint8_t>(6), static_cast<uint8_t>(7)},
                  ACL_UINT8, stream);
  RunMul<uint8_t>("TestMul_Bool_Basic_Success", {2, 3},
                  {static_cast<uint8_t>(1), static_cast<uint8_t>(0), static_cast<uint8_t>(1),
                   static_cast<uint8_t>(1), static_cast<uint8_t>(0), static_cast<uint8_t>(0)},
                  {2, 3},
                  {static_cast<uint8_t>(1), static_cast<uint8_t>(1), static_cast<uint8_t>(0),
                   static_cast<uint8_t>(1), static_cast<uint8_t>(1), static_cast<uint8_t>(0)},
                  ACL_BOOL, stream);
  RunMul<double>("TestMul_Double_Basic_Success", {2, 2}, {1.25, -2.0, 3.5, -4.5}, {2, 2}, {2.0, -3.0, 0.5, 2.0}, ACL_DOUBLE, stream);

  std::vector<uint16_t> hA = {F32ToF16(1.0f), F32ToF16(2.0f), F32ToF16(3.0f), F32ToF16(4.0f)};
  std::vector<uint16_t> hB = {F32ToF16(0.5f), F32ToF16(1.5f), F32ToF16(-2.0f), F32ToF16(3.0f)};
  RunMul<uint16_t>("TestMul_Fp16_Basic_Success", {2, 2}, hA, {2, 2}, hB, ACL_FLOAT16, stream);

  std::vector<uint16_t> bA = {F32ToBF16(1.0f), F32ToBF16(2.0f), F32ToBF16(3.0f), F32ToBF16(4.0f)};
  std::vector<uint16_t> bB = {F32ToBF16(0.25f), F32ToBF16(1.25f), F32ToBF16(-1.5f), F32ToBF16(2.0f)};
  RunMul<uint16_t>("TestMul_Bf16_Basic_Success", {2, 2}, bA, {2, 2}, bB, ACL_BF16, stream);
  RunMulMixed<uint16_t, float, float>("TestMul_Mix_Fp16xFp32_Success", {2, 2}, hA, ACL_FLOAT16,
                                      {2, 2}, {0.5f, 1.0f, -2.0f, 3.0f}, ACL_FLOAT, ACL_FLOAT, stream);
  RunMulMixed<float, uint16_t, float>("TestMul_Mix_Fp32xFp16_Success", {2, 2}, {0.5f, 1.0f, -2.0f, 3.0f}, ACL_FLOAT,
                                      {2, 2}, hA, ACL_FLOAT16, ACL_FLOAT, stream);
  RunMulMixed<uint16_t, float, float>("TestMul_Mix_Bf16xFp32_Success", {2, 2}, bA, ACL_BF16,
                                      {2, 2}, {1.5f, -0.5f, 2.0f, 0.25f}, ACL_FLOAT, ACL_FLOAT, stream);
  RunMulMixed<float, uint16_t, float>("TestMul_Mix_Fp32xBf16_Success", {2, 2}, {1.5f, -0.5f, 2.0f, 0.25f}, ACL_FLOAT,
                                      {2, 2}, bA, ACL_BF16, ACL_FLOAT, stream);

  std::vector<int64_t> c64Shape = {2, 2};
  std::vector<int64_t> c64Data = {0, 0, 0, 0};
  RunMulStatusOnly<int64_t, int64_t, int64_t>("TestMul_Complex64_StatusOnly_Success",
                                               c64Shape, c64Data, ACL_COMPLEX64, c64Shape, c64Data, ACL_COMPLEX64,
                                               c64Shape, c64Data, ACL_COMPLEX64, stream);

  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();
  RunMul<float>("TestMul_NaN_Success", {2, 2}, {nan, 1.0f, 2.0f, nan}, {2, 2}, {1.0f, nan, nan, nan}, ACL_FLOAT, stream);
  RunMul<float>("TestMul_Inf_Success", {2, 2}, {inf, -inf, 2.0f, inf}, {2, 2}, {2.0f, 2.0f, inf, 0.0f}, ACL_FLOAT, stream);

  RunMuls<float>("TestMuls_Fp32_Success", {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}, 2.5f, ACL_FLOAT, stream);
  RunMuls<uint16_t>("TestMuls_Fp16_Success", {2, 2}, hA, 1.25f, ACL_FLOAT16, stream);
  RunMuls<uint16_t>("TestMuls_Bf16_Success", {2, 2}, bA, 0.75f, ACL_BF16, stream);
  RunInplaceMul<float>("TestInplaceMul_Fp32_Broadcast_Success", {2, 3}, {1, 2, 3, 4, 5, 6}, {3}, {10, 20, 30}, ACL_FLOAT, stream);
  RunInplaceMuls<float>("TestInplaceMuls_Fp32_Success", {2, 3}, {1, 2, 3, 4, 5, 6}, 3.0f, ACL_FLOAT, stream);
  RunInplaceMuls<uint16_t>("TestInplaceMuls_Fp16_Success", {2, 2}, hA, 1.5f, ACL_FLOAT16, stream);
  RunInplaceMuls<uint16_t>("TestInplaceMuls_Bf16_Success", {2, 2}, bA, 1.5f, ACL_BF16, stream);

  RunMulExpectFail<float>("TestMul_InvalidBroadcastShape_FailExpected", {2, 3}, {1, 2, 3, 4, 5, 6}, {4, 5},
                          Repeat<float>({1.0f, 2.0f, 3.0f, 4.0f}, 20), {2, 3}, ACL_FLOAT, stream, false);
  RunMulExpectFail<float>("TestMul_InvalidOutShape_FailExpected", {2, 3}, {1, 2, 3, 4, 5, 6}, {3}, {1, 2, 3}, {2, 2},
                          ACL_FLOAT, stream, false);
  RunMulExpectFail<float>("TestMul_NullInput_FailExpected", {2, 3}, {1, 2, 3, 4, 5, 6}, {2, 3}, {1, 1, 1, 1, 1, 1}, {2, 3},
                          ACL_FLOAT, stream, true);
  RunMulExpectFail<int32_t>("TestMul_Complex32_FailExpected", {2, 2}, {0, 0, 0, 0}, {2, 2}, {0, 0, 0, 0}, {2, 2},
                            ACL_COMPLEX32, stream, false);
  RunMulsExpectFail<float>("TestMuls_InvalidOutShape_FailExpected", {2, 3}, {1, 2, 3, 4, 5, 6}, 2.0f, {3, 2},
                           ACL_FLOAT, stream, false);
  RunMulsExpectFail<float>("TestMuls_NullInput_FailExpected", {2, 3}, {1, 2, 3, 4, 5, 6}, 2.0f, {2, 3},
                           ACL_FLOAT, stream, true);
  RunInplaceMulExpectFail<float>("TestInplaceMul_InvalidBroadcast_FailExpected", {2, 3}, {1, 2, 3, 4, 5, 6},
                                 {4, 5}, Repeat<float>({1.0f, 2.0f, 3.0f}, 20), ACL_FLOAT, stream, false);
  RunInplaceMulExpectFail<float>("TestInplaceMul_NullInput_FailExpected", {2, 3}, {1, 2, 3, 4, 5, 6},
                                 {2, 3}, {1, 1, 1, 1, 1, 1}, ACL_FLOAT, stream, true);
  RunInplaceMulsExpectFail<float>("TestInplaceMuls_NullInput_FailExpected", {2, 3}, {1, 2, 3, 4, 5, 6},
                                  2.0f, ACL_FLOAT, stream, true);

  LOG_PRINT("\n========================================\n");
  LOG_PRINT("Total=%d Passed=%d Failed=%d\n", g_pass + g_fail, g_pass, g_fail);
  LOG_PRINT("========================================\n");

  Done(deviceId, stream);
  return 0;
}
