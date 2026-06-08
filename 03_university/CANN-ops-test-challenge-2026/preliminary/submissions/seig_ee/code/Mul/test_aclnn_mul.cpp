/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
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

int g_pass = 0;
int g_fail = 0;

/**
 * @brief 记录单条用例结果并更新全局计数。
 * @param name 用例名称。
 * @param ok 是否通过。
 */
void ReportCase(const char* name, bool ok) {
  if (ok) {
    printf("[PASS] %s\n", name);
    g_pass++;
  } else {
    printf("[FAIL] %s\n", name);
    g_fail++;
  }
}

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

/**
 * @brief IEEE754 float16 转 float（用于期望值与比对）。
 */
float Float16BitsToFloat(uint16_t value) {
  unsigned int sign = (value >> 15) & 0x1U;
  unsigned int exponent = (value >> 10) & 0x1fU;
  unsigned int mantissa = value & 0x3ffU;
  float result;
  if (exponent == 0U) {
    result = static_cast<float>(mantissa) * 0.0000019073486328125f;
  } else if (exponent == 31U) {
    result = (mantissa == 0U) ? std::numeric_limits<float>::infinity() : std::nanf("");
  } else {
    result = (1.0f + static_cast<float>(mantissa) * 0.0009765625f) *
             std::pow(2.0f, static_cast<float>(static_cast<int>(exponent) - 15));
  }
  return sign != 0U ? -result : result;
}

/**
 * @brief float 转 IEEE754 float16 比特（用于构造 ACL_FLOAT16 数据）。
 */
uint16_t FloatToFloat16Bits(float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
  int32_t exponent = static_cast<int32_t>(((bits >> 23) & 0xffU)) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffU;
  if (exponent <= 0) {
    return sign;
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00U);
  }
  uint16_t fp16 = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10U) | (mantissa >> 13U));
  return fp16;
}

/**
 * @brief float 转 BF16 存储（高 16 位）。
 */
uint16_t FloatToBf16Bits(float value) {
  uint32_t u = 0U;
  std::memcpy(&u, &value, sizeof(u));
  return static_cast<uint16_t>(u >> 16U);
}

float Bf16BitsToFloat(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16U;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

/**
 * @brief 将较短 shape 左侧补 1，与较长 rank 对齐。
 */
void PadShapeLeft(const std::vector<int64_t>& s, int targetRank, std::vector<int64_t>& out) {
  out.assign(targetRank, 1);
  int pad = targetRank - static_cast<int>(s.size());
  for (size_t i = 0; i < s.size(); i++) {
    out[pad + static_cast<int>(i)] = s[i];
  }
}

/**
 * @brief 推断两输入广播后的输出 shape。
 */
bool BroadcastShape(const std::vector<int64_t>& a, const std::vector<int64_t>& b, std::vector<int64_t>& out) {
  int ra = static_cast<int>(a.size());
  int rb = static_cast<int>(b.size());
  int r = std::max(ra, rb);
  std::vector<int64_t> pa(r, 1), pb(r, 1);
  PadShapeLeft(a, r, pa);
  PadShapeLeft(b, r, pb);
  out.resize(r);
  for (int i = 0; i < r; i++) {
    if (pa[i] != pb[i] && pa[i] != 1 && pb[i] != 1) {
      return false;
    }
    out[i] = std::max(pa[i], pb[i]);
  }
  return true;
}

/**
 * @brief 行主序线性下标转多维坐标。
 */
void LinearToCoord(int64_t linear, const std::vector<int64_t>& shape, std::vector<int64_t>& coord) {
  coord.resize(shape.size());
  int64_t rem = linear;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
    coord[i] = rem % shape[i];
    rem /= shape[i];
  }
}

/**
 * @brief 根据广播规则，由输出坐标得到张量线性下标。
 */
int64_t BroadcastLinearIndex(const std::vector<int64_t>& tensorShape, const std::vector<int64_t>& outShape,
                             const std::vector<int64_t>& coordOut) {
  int r = static_cast<int>(outShape.size());
  int rs = static_cast<int>(tensorShape.size());
  std::vector<int64_t> strides(rs, 1);
  for (int i = rs - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * tensorShape[i + 1];
  }
  std::vector<int64_t> ps(r, 1);
  PadShapeLeft(tensorShape, r, ps);
  int64_t idx = 0;
  for (int j = 0; j < rs; j++) {
    int pi = r - rs + j;
    int64_t c = (tensorShape[j] == 1) ? 0 : coordOut[pi];
    idx += c * strides[j];
  }
  return idx;
}

bool IsFloatType(aclDataType dt) {
  return dt == aclDataType::ACL_FLOAT || dt == aclDataType::ACL_FLOAT16 || dt == aclDataType::ACL_BF16 ||
         dt == aclDataType::ACL_DOUBLE;
}

bool IsSignedIntType(aclDataType dt) {
  return dt == aclDataType::ACL_INT8 || dt == aclDataType::ACL_INT16 || dt == aclDataType::ACL_INT32 ||
         dt == aclDataType::ACL_INT64;
}

bool IsUnsignedIntType(aclDataType dt) {
  return dt == aclDataType::ACL_UINT8;
}

/**
 * @brief 按元素类型从 host 缓冲区读取为 double（用于浮点期望或中间量）。
 */
double ReadAsDouble(const void* base, size_t idx, aclDataType dt) {
  const uint8_t* p = static_cast<const uint8_t*>(base);
  switch (dt) {
    case aclDataType::ACL_FLOAT:
      return static_cast<double>(reinterpret_cast<const float*>(p)[idx]);
    case aclDataType::ACL_DOUBLE:
      return reinterpret_cast<const double*>(p)[idx];
    case aclDataType::ACL_FLOAT16:
      return static_cast<double>(Float16BitsToFloat(reinterpret_cast<const uint16_t*>(p)[idx]));
    case aclDataType::ACL_BF16:
      return static_cast<double>(Bf16BitsToFloat(reinterpret_cast<const uint16_t*>(p)[idx]));
    case aclDataType::ACL_INT8:
      return static_cast<double>(reinterpret_cast<const int8_t*>(p)[idx]);
    case aclDataType::ACL_UINT8:
      return static_cast<double>(p[idx]);
    case aclDataType::ACL_INT16:
      return static_cast<double>(reinterpret_cast<const int16_t*>(p)[idx]);
    case aclDataType::ACL_INT32:
      return static_cast<double>(reinterpret_cast<const int32_t*>(p)[idx]);
    case aclDataType::ACL_INT64:
      return static_cast<double>(reinterpret_cast<const int64_t*>(p)[idx]);
    case aclDataType::ACL_BOOL:
      return reinterpret_cast<const uint8_t*>(p)[idx] != 0 ? 1.0 : 0.0;
    default:
      return 0.0;
  }
}

/**
 * @brief 整数乘法期望值（按输出位宽截断，与常见硬件舍入一致）。
 */
int64_t MulInt64(int64_t a, int64_t b) {
  return a * b;
}

template <typename T>
T NarrowInt(int64_t v) {
  return static_cast<T>(v);
}

/**
 * @brief 浮点容差比较：|a-b| <= atol + rtol*|b|。
 */
bool FloatClose(double actual, double expected, double atol, double rtol) {
  if (std::isnan(expected) && std::isnan(actual)) {
    return true;
  }
  if (std::isinf(expected) && std::isinf(actual) && std::signbit(expected) == std::signbit(actual)) {
    return true;
  }
  double diff = std::fabs(actual - expected);
  return diff <= atol + rtol * std::fabs(expected);
}

bool CompareOutputElem(const void* outBase, size_t idx, aclDataType outDt, double expFp, int64_t expInt, bool expBool,
                       bool useBool) {
  const uint8_t* ob = static_cast<const uint8_t*>(outBase);
  if (useBool || outDt == aclDataType::ACL_BOOL) {
    uint8_t v = reinterpret_cast<const uint8_t*>(ob)[idx];
    bool ev = expBool;
    return (v != 0) == ev;
  }
  if (IsFloatType(outDt)) {
    double actual = ReadAsDouble(outBase, idx, outDt);
    double atol = 1e-5;
    double rtol = 1e-5;
    if (outDt == aclDataType::ACL_FLOAT16) {
      atol = 1e-3;
      rtol = 1e-3;
    } else if (outDt == aclDataType::ACL_BF16) {
      atol = 1e-2;
      rtol = 1e-2;
    } else if (outDt == aclDataType::ACL_DOUBLE) {
      atol = 1e-12;
      rtol = 1e-12;
    }
    return FloatClose(actual, expFp, atol, rtol);
  }
  if (IsSignedIntType(outDt) || IsUnsignedIntType(outDt)) {
    int64_t actual = 0;
    switch (outDt) {
      case aclDataType::ACL_INT8:
        actual = reinterpret_cast<const int8_t*>(ob)[idx];
        break;
      case aclDataType::ACL_UINT8:
        actual = reinterpret_cast<const uint8_t*>(ob)[idx];
        break;
      case aclDataType::ACL_INT16:
        actual = reinterpret_cast<const int16_t*>(ob)[idx];
        break;
      case aclDataType::ACL_INT32:
        actual = reinterpret_cast<const int32_t*>(ob)[idx];
        break;
      case aclDataType::ACL_INT64:
        actual = reinterpret_cast<const int64_t*>(ob)[idx];
        break;
      default:
        return false;
    }
    return actual == expInt;
  }
  return false;
}

/**
 * @brief CPU 端广播乘法期望值。浮点以 double 计算再比对；整数用 int64 乘再截断到输出类型。
 */
bool CpuExpectMul(const void* d1, const std::vector<int64_t>& s1, aclDataType dt1, const void* d2,
                  const std::vector<int64_t>& s2, aclDataType dt2, const std::vector<int64_t>& outShape,
                  aclDataType outDt, std::vector<double>& outFp, std::vector<int64_t>& outInt,
                  std::vector<uint8_t>& outBool, bool& useFloat, bool& useInt, bool& useBool) {
  std::vector<int64_t> bshape;
  if (!BroadcastShape(s1, s2, bshape) || bshape != outShape) {
    return false;
  }
  int64_t n = GetShapeSize(outShape);
  outFp.resize(static_cast<size_t>(n));
  outInt.resize(static_cast<size_t>(n));
  outBool.resize(static_cast<size_t>(n));
  useFloat = IsFloatType(outDt);
  useInt = !useFloat && (IsSignedIntType(outDt) || IsUnsignedIntType(outDt));
  useBool = (outDt == aclDataType::ACL_BOOL);
  std::vector<int64_t> coord;
  for (int64_t i = 0; i < n; i++) {
    LinearToCoord(i, outShape, coord);
    int64_t i1 = BroadcastLinearIndex(s1, outShape, coord);
    int64_t i2 = BroadcastLinearIndex(s2, outShape, coord);
    if (useFloat) {
      outFp[static_cast<size_t>(i)] = ReadAsDouble(d1, static_cast<size_t>(i1), dt1) *
                                      ReadAsDouble(d2, static_cast<size_t>(i2), dt2);
    } else if (useBool) {
      bool a = ReadAsDouble(d1, static_cast<size_t>(i1), dt1) != 0.0;
      bool b = ReadAsDouble(d2, static_cast<size_t>(i2), dt2) != 0.0;
      outBool[static_cast<size_t>(i)] = a && b ? 1U : 0U;
    } else {
      int64_t a = static_cast<int64_t>(ReadAsDouble(d1, static_cast<size_t>(i1), dt1));
      int64_t b = static_cast<int64_t>(ReadAsDouble(d2, static_cast<size_t>(i2), dt2));
      int64_t p = MulInt64(a, b);
      switch (outDt) {
        case aclDataType::ACL_INT8:
          outInt[static_cast<size_t>(i)] = static_cast<int64_t>(NarrowInt<int8_t>(p));
          break;
        case aclDataType::ACL_UINT8:
          outInt[static_cast<size_t>(i)] = static_cast<int64_t>(NarrowInt<uint8_t>(p));
          break;
        case aclDataType::ACL_INT16:
          outInt[static_cast<size_t>(i)] = static_cast<int64_t>(NarrowInt<int16_t>(p));
          break;
        case aclDataType::ACL_INT32:
          outInt[static_cast<size_t>(i)] = static_cast<int64_t>(NarrowInt<int32_t>(p));
          break;
        case aclDataType::ACL_INT64:
          outInt[static_cast<size_t>(i)] = p;
          break;
        default:
          return false;
      }
    }
  }
  return true;
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
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
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

/**
 * @brief 两段式 API：申请 workspace、执行 phase2、同步。
 */
bool RunTwoStage(aclrtStream stream,
                 const std::function<aclnnStatus(uint64_t*, aclOpExecutor**)>& getWorkspace,
                 const std::function<aclnnStatus(void*, uint64_t, aclOpExecutor*, aclrtStream)>& runOp) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus ret = getWorkspace(&workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    return false;
  }
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      return false;
    }
  }
  ret = runOp(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    return false;
  }
  ret = aclrtSynchronizeStream(stream);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return ret == ACL_SUCCESS;
}

bool VerifyMulHost(const std::vector<int64_t>& outShape, aclDataType outDt, const void* hostOut,
                   const std::vector<double>& expFp, const std::vector<int64_t>& expInt,
                   const std::vector<uint8_t>& expBool, bool useFloat, bool useInt, bool useBool) {
  int64_t n = GetShapeSize(outShape);
  for (int64_t i = 0; i < n; i++) {
    size_t ui = static_cast<size_t>(i);
    double ef = expFp[ui];
    int64_t ei = expInt[ui];
    bool eb = expBool[ui] != 0;
    if (!CompareOutputElem(hostOut, ui, outDt, ef, ei, eb, useBool)) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return 1);

  // ---------- aclnnMul：FLOAT32 同 shape + 结果校验 ----------
  {
    const char* tag = "aclnnMul float32 same shape";
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> selfHost = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    std::vector<float> otherHost = {1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 3.f, 3.f};
    std::vector<float> outHost(static_cast<size_t>(GetShapeSize(shape)), 0.f);
    void* daSelf = nullptr;
    void* daOther = nullptr;
    void* daOut = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    aclTensor* out = nullptr;
    bool ok = true;
    ret = CreateAclTensor(selfHost, shape, &daSelf, aclDataType::ACL_FLOAT, &self);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(otherHost, shape, &daOther, aclDataType::ACL_FLOAT, &other);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(outHost, shape, &daOut, aclDataType::ACL_FLOAT, &out);
    ok = ok && (ret == 0);
    if (ok) {
      ok = RunTwoStage(
          stream,
          [&](uint64_t* ws, aclOpExecutor** ex) { return aclnnMulGetWorkspaceSize(self, other, out, ws, ex); },
          [&](void* w, uint64_t sz, aclOpExecutor* ex, aclrtStream st) { return aclnnMul(w, sz, ex, st); });
    }
    std::vector<double> expFp;
    std::vector<int64_t> expInt;
    std::vector<uint8_t> expBool;
    bool uf = false, ui = false, ub = false;
    CpuExpectMul(selfHost.data(), shape, aclDataType::ACL_FLOAT, otherHost.data(), shape, aclDataType::ACL_FLOAT, shape,
                 aclDataType::ACL_FLOAT, expFp, expInt, expBool, uf, ui, ub);
    std::vector<float> result(static_cast<size_t>(GetShapeSize(shape)));
    if (ok) {
      ret = aclrtMemcpy(result.data(), result.size() * sizeof(float), daOut, result.size() * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      ok = (ret == ACL_SUCCESS) && VerifyMulHost(shape, aclDataType::ACL_FLOAT, result.data(), expFp, expInt, expBool,
                                                  true, false, false);
    }
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    aclrtFree(daSelf);
    aclrtFree(daOther);
    aclrtFree(daOut);
    ReportCase(tag, ok);
  }

  // ---------- aclnnMul：广播 [2,3] * [3] ----------
  {
    const char* tag = "aclnnMul float32 broadcast [2,3] x [3]";
    std::vector<int64_t> s1 = {2, 3};
    std::vector<int64_t> s2 = {3};
    std::vector<int64_t> outShape;
    BroadcastShape(s1, s2, outShape);
    std::vector<float> h1 = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<float> h2 = {10.f, 20.f, 30.f};
    std::vector<float> outInit(static_cast<size_t>(GetShapeSize(outShape)), 0.f);
    void* d1 = nullptr;
    void* d2 = nullptr;
    void* dout = nullptr;
    aclTensor* t1 = nullptr;
    aclTensor* t2 = nullptr;
    aclTensor* tout = nullptr;
    bool ok = true;
    ret = CreateAclTensor(h1, s1, &d1, aclDataType::ACL_FLOAT, &t1);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(h2, s2, &d2, aclDataType::ACL_FLOAT, &t2);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(outInit, outShape, &dout, aclDataType::ACL_FLOAT, &tout);
    ok = ok && (ret == 0);
    if (ok) {
      ok = RunTwoStage(stream,
                       [&](uint64_t* ws, aclOpExecutor** ex) { return aclnnMulGetWorkspaceSize(t1, t2, tout, ws, ex); },
                       [&](void* w, uint64_t sz, aclOpExecutor* ex, aclrtStream st) { return aclnnMul(w, sz, ex, st); });
    }
    std::vector<double> expFp;
    std::vector<int64_t> expInt;
    std::vector<uint8_t> expBool;
    bool uf = false, ui = false, ub = false;
    CpuExpectMul(h1.data(), s1, aclDataType::ACL_FLOAT, h2.data(), s2, aclDataType::ACL_FLOAT, outShape,
                 aclDataType::ACL_FLOAT, expFp, expInt, expBool, uf, ui, ub);
    std::vector<float> res(static_cast<size_t>(GetShapeSize(outShape)));
    if (ok) {
      ret = aclrtMemcpy(res.data(), res.size() * sizeof(float), dout, res.size() * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      ok = (ret == ACL_SUCCESS) && VerifyMulHost(outShape, aclDataType::ACL_FLOAT, res.data(), expFp, expInt, expBool,
                                                 true, false, false);
    }
    aclDestroyTensor(t1);
    aclDestroyTensor(t2);
    aclDestroyTensor(tout);
    aclrtFree(d1);
    aclrtFree(d2);
    aclrtFree(dout);
    ReportCase(tag, ok);
  }

  // ---------- aclnnMul：FLOAT16 + FLOAT -> FLOAT 混合 ----------
  {
    const char* tag = "aclnnMul mixed fp16 x fp32 -> fp32";
    std::vector<int64_t> shape = {2, 2};
    std::vector<uint16_t> h1 = {FloatToFloat16Bits(1.f), FloatToFloat16Bits(2.f), FloatToFloat16Bits(3.f),
                                FloatToFloat16Bits(4.f)};
    std::vector<float> h2 = {2.f, 2.f, 2.f, 2.f};
    std::vector<float> out0(static_cast<size_t>(GetShapeSize(shape)), 0.f);
    void* d1 = nullptr;
    void* d2 = nullptr;
    void* dout = nullptr;
    aclTensor* t1 = nullptr;
    aclTensor* t2 = nullptr;
    aclTensor* tout = nullptr;
    bool ok = true;
    ret = CreateAclTensor(h1, shape, &d1, aclDataType::ACL_FLOAT16, &t1);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(h2, shape, &d2, aclDataType::ACL_FLOAT, &t2);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(out0, shape, &dout, aclDataType::ACL_FLOAT, &tout);
    ok = ok && (ret == 0);
    if (ok) {
      ok = RunTwoStage(stream,
                       [&](uint64_t* ws, aclOpExecutor** ex) { return aclnnMulGetWorkspaceSize(t1, t2, tout, ws, ex); },
                       [&](void* w, uint64_t sz, aclOpExecutor* ex, aclrtStream st) { return aclnnMul(w, sz, ex, st); });
    }
    std::vector<double> expFp;
    std::vector<int64_t> expInt;
    std::vector<uint8_t> expBool;
    bool uf = false, ui = false, ub = false;
    CpuExpectMul(h1.data(), shape, aclDataType::ACL_FLOAT16, h2.data(), shape, aclDataType::ACL_FLOAT, shape,
                 aclDataType::ACL_FLOAT, expFp, expInt, expBool, uf, ui, ub);
    std::vector<float> res(static_cast<size_t>(GetShapeSize(shape)));
    if (ok) {
      ret = aclrtMemcpy(res.data(), res.size() * sizeof(float), dout, res.size() * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      ok = (ret == ACL_SUCCESS) && VerifyMulHost(shape, aclDataType::ACL_FLOAT, res.data(), expFp, expInt, expBool,
                                                 true, false, false);
    }
    aclDestroyTensor(t1);
    aclDestroyTensor(t2);
    aclDestroyTensor(tout);
    aclrtFree(d1);
    aclrtFree(d2);
    aclrtFree(dout);
    ReportCase(tag, ok);
  }

  // ---------- aclnnMul：BF16 * FLOAT -> FLOAT ----------
  {
    const char* tag = "aclnnMul mixed bf16 x fp32 -> fp32";
    std::vector<int64_t> shape = {1, 4};
    std::vector<uint16_t> h1 = {FloatToBf16Bits(1.f), FloatToBf16Bits(-2.f), FloatToBf16Bits(0.f),
                                 FloatToBf16Bits(0.25f)};
    std::vector<float> h2 = {4.f, 3.f, 2.f, 8.f};
    std::vector<float> out0(static_cast<size_t>(GetShapeSize(shape)), 0.f);
    void* d1 = nullptr;
    void* d2 = nullptr;
    void* dout = nullptr;
    aclTensor* t1 = nullptr;
    aclTensor* t2 = nullptr;
    aclTensor* tout = nullptr;
    bool ok = true;
    ret = CreateAclTensor(h1, shape, &d1, aclDataType::ACL_BF16, &t1);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(h2, shape, &d2, aclDataType::ACL_FLOAT, &t2);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(out0, shape, &dout, aclDataType::ACL_FLOAT, &tout);
    ok = ok && (ret == 0);
    if (ok) {
      ok = RunTwoStage(stream,
                       [&](uint64_t* ws, aclOpExecutor** ex) { return aclnnMulGetWorkspaceSize(t1, t2, tout, ws, ex); },
                       [&](void* w, uint64_t sz, aclOpExecutor* ex, aclrtStream st) { return aclnnMul(w, sz, ex, st); });
    }
    std::vector<double> expFp;
    std::vector<int64_t> expInt;
    std::vector<uint8_t> expBool;
    bool uf = false, ui = false, ub = false;
    CpuExpectMul(h1.data(), shape, aclDataType::ACL_BF16, h2.data(), shape, aclDataType::ACL_FLOAT, shape,
                 aclDataType::ACL_FLOAT, expFp, expInt, expBool, uf, ui, ub);
    std::vector<float> res(static_cast<size_t>(GetShapeSize(shape)));
    if (ok) {
      ret = aclrtMemcpy(res.data(), res.size() * sizeof(float), dout, res.size() * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      ok = (ret == ACL_SUCCESS) && VerifyMulHost(shape, aclDataType::ACL_FLOAT, res.data(), expFp, expInt, expBool,
                                                 true, false, false);
    }
    aclDestroyTensor(t1);
    aclDestroyTensor(t2);
    aclDestroyTensor(tout);
    aclrtFree(d1);
    aclrtFree(d2);
    aclrtFree(dout);
    ReportCase(tag, ok);
  }

  // ---------- aclnnMul：INT32 ----------
  {
    const char* tag = "aclnnMul int32";
    std::vector<int64_t> shape = {3};
    std::vector<int32_t> h1 = {1000, -2, 0};
    std::vector<int32_t> h2 = {2, 3, 999};
    std::vector<int32_t> out0(static_cast<size_t>(GetShapeSize(shape)), 0);
    void* d1 = nullptr;
    void* d2 = nullptr;
    void* dout = nullptr;
    aclTensor* t1 = nullptr;
    aclTensor* t2 = nullptr;
    aclTensor* tout = nullptr;
    bool ok = true;
    ret = CreateAclTensor(h1, shape, &d1, aclDataType::ACL_INT32, &t1);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(h2, shape, &d2, aclDataType::ACL_INT32, &t2);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(out0, shape, &dout, aclDataType::ACL_INT32, &tout);
    ok = ok && (ret == 0);
    if (ok) {
      ok = RunTwoStage(stream,
                       [&](uint64_t* ws, aclOpExecutor** ex) { return aclnnMulGetWorkspaceSize(t1, t2, tout, ws, ex); },
                       [&](void* w, uint64_t sz, aclOpExecutor* ex, aclrtStream st) { return aclnnMul(w, sz, ex, st); });
    }
    std::vector<double> expFp;
    std::vector<int64_t> expInt;
    std::vector<uint8_t> expBool;
    bool uf = false, ui = false, ub = false;
    CpuExpectMul(h1.data(), shape, aclDataType::ACL_INT32, h2.data(), shape, aclDataType::ACL_INT32, shape,
                 aclDataType::ACL_INT32, expFp, expInt, expBool, uf, ui, ub);
    std::vector<int32_t> res(static_cast<size_t>(GetShapeSize(shape)));
    if (ok) {
      ret = aclrtMemcpy(res.data(), res.size() * sizeof(int32_t), dout, res.size() * sizeof(int32_t),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      ok = (ret == ACL_SUCCESS) && VerifyMulHost(shape, aclDataType::ACL_INT32, res.data(), expFp, expInt, expBool,
                                                 false, true, false);
    }
    aclDestroyTensor(t1);
    aclDestroyTensor(t2);
    aclDestroyTensor(tout);
    aclrtFree(d1);
    aclrtFree(d2);
    aclrtFree(dout);
    ReportCase(tag, ok);
  }

  // ---------- aclnnMul：INT8 广播 ----------
  {
    const char* tag = "aclnnMul int8 broadcast";
    std::vector<int64_t> s1 = {2, 1};
    std::vector<int64_t> s2 = {1, 4};
    std::vector<int64_t> osh;
    BroadcastShape(s1, s2, osh);
    std::vector<int8_t> h1 = {3, -1};
    std::vector<int8_t> h2 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int8_t> out0(static_cast<size_t>(GetShapeSize(osh)), 0);
    void* d1 = nullptr;
    void* d2 = nullptr;
    void* dout = nullptr;
    aclTensor* t1 = nullptr;
    aclTensor* t2 = nullptr;
    aclTensor* tout = nullptr;
    bool ok = true;
    ret = CreateAclTensor(h1, s1, &d1, aclDataType::ACL_INT8, &t1);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(h2, s2, &d2, aclDataType::ACL_INT8, &t2);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(out0, osh, &dout, aclDataType::ACL_INT8, &tout);
    ok = ok && (ret == 0);
    if (ok) {
      ok = RunTwoStage(stream,
                       [&](uint64_t* ws, aclOpExecutor** ex) { return aclnnMulGetWorkspaceSize(t1, t2, tout, ws, ex); },
                       [&](void* w, uint64_t sz, aclOpExecutor* ex, aclrtStream st) { return aclnnMul(w, sz, ex, st); });
    }
    std::vector<double> expFp;
    std::vector<int64_t> expInt;
    std::vector<uint8_t> expBool;
    bool uf = false, ui = false, ub = false;
    CpuExpectMul(h1.data(), s1, aclDataType::ACL_INT8, h2.data(), s2, aclDataType::ACL_INT8, osh, aclDataType::ACL_INT8,
                 expFp, expInt, expBool, uf, ui, ub);
    std::vector<int8_t> res(static_cast<size_t>(GetShapeSize(osh)));
    if (ok) {
      ret = aclrtMemcpy(res.data(), res.size(), dout, res.size(), ACL_MEMCPY_DEVICE_TO_HOST);
      ok = (ret == ACL_SUCCESS) && VerifyMulHost(osh, aclDataType::ACL_INT8, res.data(), expFp, expInt, expBool, false,
                                                 true, false);
    }
    aclDestroyTensor(t1);
    aclDestroyTensor(t2);
    aclDestroyTensor(tout);
    aclrtFree(d1);
    aclrtFree(d2);
    aclrtFree(dout);
    ReportCase(tag, ok);
  }

  // ---------- aclnnMul：DOUBLE 边界（含 Inf） ----------
  {
    const char* tag = "aclnnMul double inf/nan";
    std::vector<int64_t> shape = {3};
    double pInf = std::numeric_limits<double>::infinity();
    double nInf = -pInf;
    std::vector<double> h1 = {2.0, 0.0, nInf};
    std::vector<double> h2 = {pInf, 3.0, 0.0};
    std::vector<double> out0(static_cast<size_t>(GetShapeSize(shape)), 0.0);
    void* d1 = nullptr;
    void* d2 = nullptr;
    void* dout = nullptr;
    aclTensor* t1 = nullptr;
    aclTensor* t2 = nullptr;
    aclTensor* tout = nullptr;
    bool ok = true;
    ret = CreateAclTensor(h1, shape, &d1, aclDataType::ACL_DOUBLE, &t1);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(h2, shape, &d2, aclDataType::ACL_DOUBLE, &t2);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(out0, shape, &dout, aclDataType::ACL_DOUBLE, &tout);
    ok = ok && (ret == 0);
    if (ok) {
      ok = RunTwoStage(stream,
                       [&](uint64_t* ws, aclOpExecutor** ex) { return aclnnMulGetWorkspaceSize(t1, t2, tout, ws, ex); },
                       [&](void* w, uint64_t sz, aclOpExecutor* ex, aclrtStream st) { return aclnnMul(w, sz, ex, st); });
    }
    std::vector<double> expFp;
    std::vector<int64_t> expInt;
    std::vector<uint8_t> expBool;
    bool uf = false, ui = false, ub = false;
    CpuExpectMul(h1.data(), shape, aclDataType::ACL_DOUBLE, h2.data(), shape, aclDataType::ACL_DOUBLE, shape,
                 aclDataType::ACL_DOUBLE, expFp, expInt, expBool, uf, ui, ub);
    std::vector<double> res(static_cast<size_t>(GetShapeSize(shape)));
    if (ok) {
      ret = aclrtMemcpy(res.data(), res.size() * sizeof(double), dout, res.size() * sizeof(double),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      ok = (ret == ACL_SUCCESS) && VerifyMulHost(shape, aclDataType::ACL_DOUBLE, res.data(), expFp, expInt, expBool,
                                                 true, false, false);
    }
    aclDestroyTensor(t1);
    aclDestroyTensor(t2);
    aclDestroyTensor(tout);
    aclrtFree(d1);
    aclrtFree(d2);
    aclrtFree(dout);
    ReportCase(tag, ok);
  }

  // ---------- aclnnMuls：标量乘 ----------
  {
    const char* tag = "aclnnMuls float tensor x scalar";
    std::vector<int64_t> shape = {2, 3};
    std::vector<float> hself = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float sc = -0.5f;
    std::vector<float> out0(static_cast<size_t>(GetShapeSize(shape)), 0.f);
    void* ds = nullptr;
    void* dout = nullptr;
    aclTensor* self = nullptr;
    aclScalar* scl = nullptr;
    aclTensor* out = nullptr;
    bool ok = true;
    ret = CreateAclTensor(hself, shape, &ds, aclDataType::ACL_FLOAT, &self);
    ok = ok && (ret == 0);
    scl = aclCreateScalar(&sc, aclDataType::ACL_FLOAT);
    ok = ok && (scl != nullptr);
    ret = CreateAclTensor(out0, shape, &dout, aclDataType::ACL_FLOAT, &out);
    ok = ok && (ret == 0);
    if (ok) {
      ok = RunTwoStage(stream,
                       [&](uint64_t* ws, aclOpExecutor** ex) { return aclnnMulsGetWorkspaceSize(self, scl, out, ws, ex); },
                       [&](void* w, uint64_t sz, aclOpExecutor* ex, aclrtStream st) { return aclnnMuls(w, sz, ex, st); });
    }
    std::vector<float> otherExpanded(hself.size());
    for (size_t i = 0; i < hself.size(); i++) {
      otherExpanded[i] = sc;
    }
    std::vector<double> expFp;
    std::vector<int64_t> expInt;
    std::vector<uint8_t> expBool;
    bool uf = false, ui = false, ub = false;
    CpuExpectMul(hself.data(), shape, aclDataType::ACL_FLOAT, otherExpanded.data(), shape, aclDataType::ACL_FLOAT, shape,
                 aclDataType::ACL_FLOAT, expFp, expInt, expBool, uf, ui, ub);
    std::vector<float> res(static_cast<size_t>(GetShapeSize(shape)));
    if (ok) {
      ret = aclrtMemcpy(res.data(), res.size() * sizeof(float), dout, res.size() * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      ok = (ret == ACL_SUCCESS) && VerifyMulHost(shape, aclDataType::ACL_FLOAT, res.data(), expFp, expInt, expBool,
                                                 true, false, false);
    }
    aclDestroyTensor(self);
    if (scl) {
      aclDestroyScalar(scl);
    }
    aclDestroyTensor(out);
    aclrtFree(ds);
    aclrtFree(dout);
    ReportCase(tag, ok);
  }

  // ---------- aclnnInplaceMul ----------
  {
    const char* tag = "aclnnInplaceMul float32";
    std::vector<int64_t> shape = {2, 2};
    std::vector<float> hs = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> ho = {2.f, 0.5f, -1.f, 2.f};
    void* ds = nullptr;
    void* doo = nullptr;
    aclTensor* self = nullptr;
    aclTensor* other = nullptr;
    bool ok = true;
    ret = CreateAclTensor(hs, shape, &ds, aclDataType::ACL_FLOAT, &self);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(ho, shape, &doo, aclDataType::ACL_FLOAT, &other);
    ok = ok && (ret == 0);
    if (ok) {
      ok = RunTwoStage(
          stream,
          [&](uint64_t* ws, aclOpExecutor** ex) { return aclnnInplaceMulGetWorkspaceSize(self, other, ws, ex); },
          [&](void* w, uint64_t sz, aclOpExecutor* ex, aclrtStream st) { return aclnnInplaceMul(w, sz, ex, st); });
    }
    std::vector<double> expFp;
    std::vector<int64_t> expInt;
    std::vector<uint8_t> expBool;
    bool uf = false, ui = false, ub = false;
    CpuExpectMul(hs.data(), shape, aclDataType::ACL_FLOAT, ho.data(), shape, aclDataType::ACL_FLOAT, shape,
                 aclDataType::ACL_FLOAT, expFp, expInt, expBool, uf, ui, ub);
    std::vector<float> res(static_cast<size_t>(GetShapeSize(shape)));
    if (ok) {
      ret = aclrtMemcpy(res.data(), res.size() * sizeof(float), ds, res.size() * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      ok = (ret == ACL_SUCCESS) && VerifyMulHost(shape, aclDataType::ACL_FLOAT, res.data(), expFp, expInt, expBool,
                                                 true, false, false);
    }
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclrtFree(ds);
    aclrtFree(doo);
    ReportCase(tag, ok);
  }

  // ---------- aclnnInplaceMuls ----------
  {
    const char* tag = "aclnnInplaceMuls int32";
    std::vector<int64_t> shape = {4};
    std::vector<int32_t> hs = {10, -2, 7, 0};
    int32_t sc = 3;
    void* ds = nullptr;
    aclTensor* self = nullptr;
    aclScalar* scl = nullptr;
    bool ok = true;
    ret = CreateAclTensor(hs, shape, &ds, aclDataType::ACL_INT32, &self);
    ok = ok && (ret == 0);
    scl = aclCreateScalar(&sc, aclDataType::ACL_INT32);
    ok = ok && (scl != nullptr);
    if (ok) {
      ok = RunTwoStage(stream,
                       [&](uint64_t* ws, aclOpExecutor** ex) {
                         return aclnnInplaceMulsGetWorkspaceSize(self, scl, ws, ex);
                       },
                       [&](void* w, uint64_t sz, aclOpExecutor* ex, aclrtStream st) {
                         return aclnnInplaceMuls(w, sz, ex, st);
                       });
    }
    std::vector<int32_t> expanded(hs.size(), sc);
    std::vector<double> expFp;
    std::vector<int64_t> expInt;
    std::vector<uint8_t> expBool;
    bool uf = false, ui = false, ub = false;
    CpuExpectMul(hs.data(), shape, aclDataType::ACL_INT32, expanded.data(), shape, aclDataType::ACL_INT32, shape,
                 aclDataType::ACL_INT32, expFp, expInt, expBool, uf, ui, ub);
    std::vector<int32_t> res(static_cast<size_t>(GetShapeSize(shape)));
    if (ok) {
      ret = aclrtMemcpy(res.data(), res.size() * sizeof(int32_t), ds, res.size() * sizeof(int32_t),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      ok = (ret == ACL_SUCCESS) && VerifyMulHost(shape, aclDataType::ACL_INT32, res.data(), expFp, expInt, expBool,
                                                 false, true, false);
    }
    aclDestroyTensor(self);
    if (scl) {
      aclDestroyScalar(scl);
    }
    aclrtFree(ds);
    ReportCase(tag, ok);
  }

  // ---------- 异常：nullptr 应失败 ----------
  {
    const char* tag = "aclnnMul nullptr expect failure";
    uint64_t ws = 0;
    aclOpExecutor* ex = nullptr;
    std::vector<int64_t> sh = {1};
    std::vector<float> one = {1.f};
    void* d1 = nullptr;
    void* d2 = nullptr;
    aclTensor* t1 = nullptr;
    aclTensor* t2 = nullptr;
    const int c1 = CreateAclTensor(one, sh, &d1, aclDataType::ACL_FLOAT, &t1);
    const int c2 = (c1 == 0) ? CreateAclTensor(one, sh, &d2, aclDataType::ACL_FLOAT, &t2) : -1;
    bool ok = false;
    if (c1 == 0 && c2 == 0) {
      aclnnStatus st = aclnnMulGetWorkspaceSize(nullptr, t2, t1, &ws, &ex);
      ok = (st != ACL_SUCCESS);
    }
    if (t1 != nullptr) {
      aclDestroyTensor(t1);
    }
    if (t2 != nullptr) {
      aclDestroyTensor(t2);
    }
    if (d1 != nullptr) {
      aclrtFree(d1);
    }
    if (d2 != nullptr) {
      aclrtFree(d2);
    }
    ReportCase(tag, ok);
  }

  // ---------- 较大 tensor（tiling 路径） ----------
  {
    const char* tag = "aclnnMul large float tensor";
    const int64_t n = 4096;
    std::vector<int64_t> shape = {n};
    std::vector<float> h1(static_cast<size_t>(n));
    std::vector<float> h2(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) {
      h1[static_cast<size_t>(i)] = static_cast<float>(i % 17) * 0.25f;
      h2[static_cast<size_t>(i)] = static_cast<float>((i * 3) % 31) * 0.1f;
    }
    std::vector<float> out0(static_cast<size_t>(n), 0.f);
    void* d1 = nullptr;
    void* d2 = nullptr;
    void* dout = nullptr;
    aclTensor* t1 = nullptr;
    aclTensor* t2 = nullptr;
    aclTensor* tout = nullptr;
    bool ok = true;
    ret = CreateAclTensor(h1, shape, &d1, aclDataType::ACL_FLOAT, &t1);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(h2, shape, &d2, aclDataType::ACL_FLOAT, &t2);
    ok = ok && (ret == 0);
    ret = CreateAclTensor(out0, shape, &dout, aclDataType::ACL_FLOAT, &tout);
    ok = ok && (ret == 0);
    if (ok) {
      ok = RunTwoStage(stream,
                       [&](uint64_t* ws, aclOpExecutor** ex) { return aclnnMulGetWorkspaceSize(t1, t2, tout, ws, ex); },
                       [&](void* w, uint64_t sz, aclOpExecutor* ex, aclrtStream st) { return aclnnMul(w, sz, ex, st); });
    }
    std::vector<double> expFp;
    std::vector<int64_t> expInt;
    std::vector<uint8_t> expBool;
    bool uf = false, ui = false, ub = false;
    CpuExpectMul(h1.data(), shape, aclDataType::ACL_FLOAT, h2.data(), shape, aclDataType::ACL_FLOAT, shape,
                 aclDataType::ACL_FLOAT, expFp, expInt, expBool, uf, ui, ub);
    std::vector<float> res(static_cast<size_t>(n));
    if (ok) {
      ret = aclrtMemcpy(res.data(), res.size() * sizeof(float), dout, res.size() * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      ok = (ret == ACL_SUCCESS) && VerifyMulHost(shape, aclDataType::ACL_FLOAT, res.data(), expFp, expInt, expBool,
                                                 true, false, false);
    }
    aclDestroyTensor(t1);
    aclDestroyTensor(t2);
    aclDestroyTensor(tout);
    aclrtFree(d1);
    aclrtFree(d2);
    aclrtFree(dout);
    ReportCase(tag, ok);
  }

  printf("Summary: PASS=%d FAIL=%d\n", g_pass, g_fail);
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return g_fail > 0 ? 1 : 0;
}
