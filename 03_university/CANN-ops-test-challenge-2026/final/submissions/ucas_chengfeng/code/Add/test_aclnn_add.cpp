/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if __has_include("aclnnop/aclnn_add.h")
#include "aclnnop/aclnn_add.h"
#else
#include "../op_api/aclnn_add.h"
#endif

#if __has_include("aclnnop/aclnn_add_v3.h")
#include "aclnnop/aclnn_add_v3.h"
#else
#include "../op_api/aclnn_add_v3.h"
#endif

#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif

#ifndef ACLNN_ERR_PARAM_NULLPTR
#define ACLNN_ERR_PARAM_NULLPTR 161001
#endif

#ifndef ACLNN_ERR_PARAM_INVALID
#define ACLNN_ERR_PARAM_INVALID 161002
#endif

#define CHECK_RET(cond, return_expr) \
  do                                 \
  {                                  \
    if (!(cond))                     \
    {                                \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do                                \
  {                                 \
    printf(message, ##__VA_ARGS__); \
  } while (0)

namespace
{

  enum class Phase
  {
    kApi,
    kExec,
  };

  enum class ApiKind
  {
    kAdd,
    kAdds,
    kInplaceAdd,
    kInplaceAdds,
    kAddV3,
    kInplaceAddV3,
  };

  enum class DType
  {
    kFloat,
    kFloat16,
    kBFloat16,
    kDouble,
    kInt32,
    kInt64,
    kInt16,
    kInt8,
    kUInt8,
    kBool,
    kComplex32,
    kComplex64,
  };

  struct Tolerance
  {
    double atol = 0.0;
    double rtol = 0.0;
  };

  struct Options
  {
    std::string phase = "all";
    std::string caseId;
    int32_t deviceId = 0;
    bool includeBroadcastProbes = false;
  };

  struct CaseSpec
  {
    std::string caseId;
    std::string title;
    ApiKind api = ApiKind::kAdd;
    DType lhsDType = DType::kFloat;
    DType rhsDType = DType::kFloat;
    DType alphaDType = DType::kFloat;
    DType outDType = DType::kFloat;
    std::vector<int64_t> lhsShape;
    std::vector<int64_t> rhsShape;
    std::vector<int64_t> outShape;
    std::vector<std::string> lhsData;
    std::vector<std::string> rhsData;
    std::vector<std::string> alphaData;
    int expectStatus = ACLNN_SUCCESS;
    std::string nullWhich;
    bool allowSkip = false;
    std::string skipReason;
    Tolerance tolerance;
  };

  struct DeviceTensor
  {
    void *deviceAddr = nullptr;
    aclTensor *tensor = nullptr;
  };

  struct HostScalar
  {
    std::vector<uint8_t> storage;
    aclScalar *scalar = nullptr;
  };

  struct RuntimeContext
  {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    std::string socName;
  };

  struct CaseResult
  {
    std::string caseId;
    std::string title;
    std::string phase;
    std::string api;
    int statusCode = ACLNN_SUCCESS;
    bool pass = false;
    bool skipped = false;
    std::string message;
    std::vector<std::string> expected;
    std::vector<std::string> actual;
    int64_t firstMismatchIndex = -1;
    double maxAbsError = 0.0;
    double maxRelError = 0.0;
  };

  std::string JoinTokens(const std::vector<std::string> &tokens)
  {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < tokens.size(); ++i)
    {
      if (i > 0)
      {
        oss << ", ";
      }
      oss << tokens[i];
    }
    oss << "]";
    return oss.str();
  }

  using RefReal = long double;
  using RefComplex = std::complex<RefReal>;

  struct ScalarToken
  {
    enum class Kind
    {
      kFinite,
      kPosInf,
      kNegInf,
      kNaN,
      kBool,
    };

    Kind kind = Kind::kFinite;
    RefReal value = 0.0L;
    bool boolValue = false;
    bool negativeZero = false;
  };

  struct RefValue
  {
    ScalarToken::Kind kind = ScalarToken::Kind::kFinite;
    RefReal value = 0.0L;
    bool boolValue = false;
    bool negativeZero = false;
    bool isComplex = false;
    RefComplex complexValue{0.0L, 0.0L};
  };

  constexpr int kAclnnInnerNullPtr = 561103;

  ScalarToken ParseScalarToken(const std::string &token);

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
    if (shape.empty())
    {
      return 1;
    }
    int64_t size = 1;
    for (int64_t dim : shape)
    {
      size *= dim;
    }
    return size;
  }

  std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &shape)
  {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i)
    {
      strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
  }

  std::vector<int64_t> InferBroadcastShape(const std::vector<int64_t> &lhs, const std::vector<int64_t> &rhs)
  {
    const size_t outRank = std::max(lhs.size(), rhs.size());
    std::vector<int64_t> out(outRank, 1);
    for (size_t i = 0; i < outRank; ++i)
    {
      const int64_t lhsDim = i < outRank - lhs.size() ? 1 : lhs[i - (outRank - lhs.size())];
      const int64_t rhsDim = i < outRank - rhs.size() ? 1 : rhs[i - (outRank - rhs.size())];
      if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1)
      {
        throw std::runtime_error("broadcast shape inference failed");
      }
      out[i] = std::max(lhsDim, rhsDim);
    }
    return out;
  }

  std::string ToLower(std::string value)
  {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch)
                   { return static_cast<char>(std::tolower(ch)); });
    return value;
  }

  bool IsFloatingType(DType dtype)
  {
    return dtype == DType::kFloat || dtype == DType::kFloat16 || dtype == DType::kBFloat16 || dtype == DType::kDouble;
  }

  bool IsComplexType(DType dtype)
  {
    return dtype == DType::kComplex32 || dtype == DType::kComplex64;
  }

  std::string DTypeToString(DType dtype)
  {
    switch (dtype)
    {
    case DType::kFloat:
      return "float32";
    case DType::kFloat16:
      return "float16";
    case DType::kBFloat16:
      return "bfloat16";
    case DType::kDouble:
      return "float64";
    case DType::kInt32:
      return "int32";
    case DType::kInt64:
      return "int64";
    case DType::kInt16:
      return "int16";
    case DType::kInt8:
      return "int8";
    case DType::kUInt8:
      return "uint8";
    case DType::kBool:
      return "bool";
    case DType::kComplex32:
      return "complex32";
    case DType::kComplex64:
      return "complex64";
    }
    return "unknown";
  }

  std::string ApiKindToString(ApiKind api)
  {
    switch (api)
    {
    case ApiKind::kAdd:
      return "Add";
    case ApiKind::kAdds:
      return "Adds";
    case ApiKind::kInplaceAdd:
      return "InplaceAdd";
    case ApiKind::kInplaceAdds:
      return "InplaceAdds";
    case ApiKind::kAddV3:
      return "AddV3";
    case ApiKind::kInplaceAddV3:
      return "InplaceAddV3";
    }
    return "unknown";
  }

  bool IsInplace(ApiKind api)
  {
    return api == ApiKind::kInplaceAdd || api == ApiKind::kInplaceAdds || api == ApiKind::kInplaceAddV3;
  }

  bool LhsIsScalar(ApiKind api)
  {
    return api == ApiKind::kAddV3 || api == ApiKind::kInplaceAddV3;
  }

  bool RhsIsScalar(ApiKind api)
  {
    return api == ApiKind::kAdds || api == ApiKind::kInplaceAdds;
  }

  size_t DTypeSize(DType dtype)
  {
    switch (dtype)
    {
    case DType::kFloat:
      return sizeof(float);
    case DType::kFloat16:
      return sizeof(aclFloat16);
    case DType::kBFloat16:
      return sizeof(uint16_t);
    case DType::kDouble:
      return sizeof(double);
    case DType::kInt32:
      return sizeof(int32_t);
    case DType::kInt64:
      return sizeof(int64_t);
    case DType::kInt16:
      return sizeof(int16_t);
    case DType::kInt8:
      return sizeof(int8_t);
    case DType::kUInt8:
      return sizeof(uint8_t);
    case DType::kBool:
      return sizeof(bool);
    case DType::kComplex32:
      return sizeof(aclFloat16) * 2;
    case DType::kComplex64:
      return sizeof(float) * 2;
    }
    return 0;
  }

  aclDataType ToAclDataType(DType dtype)
  {
    switch (dtype)
    {
    case DType::kFloat:
      return ACL_FLOAT;
    case DType::kFloat16:
      return ACL_FLOAT16;
    case DType::kBFloat16:
      return ACL_BF16;
    case DType::kDouble:
      return ACL_DOUBLE;
    case DType::kInt32:
      return ACL_INT32;
    case DType::kInt64:
      return ACL_INT64;
    case DType::kInt16:
      return ACL_INT16;
    case DType::kInt8:
      return ACL_INT8;
    case DType::kUInt8:
      return ACL_UINT8;
    case DType::kBool:
      return ACL_BOOL;
    case DType::kComplex32:
      return ACL_COMPLEX32;
    case DType::kComplex64:
      return ACL_COMPLEX64;
    }
    return ACL_DT_UNDEFINED;
  }

  Tolerance DefaultTolerance(DType dtype)
  {
    switch (dtype)
    {
    case DType::kFloat16:
      return {1e-4, 1e-4};
    case DType::kBFloat16:
      return {1e-2, 1e-2};
    case DType::kFloat:
      return {1e-6, 1e-6};
    case DType::kDouble:
      return {1e-12, 1e-12};
    case DType::kComplex32:
      return {1e-4, 1e-4};
    case DType::kComplex64:
      return {1e-6, 1e-6};
    default:
      return {0.0, 0.0};
    }
  }

  std::pair<ScalarToken, ScalarToken> ParseComplexToken(const std::string &token)
  {
    const size_t comma = token.find(',');
    CHECK_RET(comma != std::string::npos, throw std::runtime_error("invalid complex token: " + token));
    return {ParseScalarToken(token.substr(0, comma)), ParseScalarToken(token.substr(comma + 1))};
  }

  template <typename T>
  void AppendBytes(std::vector<uint8_t> *buffer, const T &value)
  {
    const uint8_t *src = reinterpret_cast<const uint8_t *>(&value);
    buffer->insert(buffer->end(), src, src + sizeof(T));
  }

  uint16_t FloatToBFloat16(float value)
  {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7FFFU + lsb;
    return static_cast<uint16_t>(bits >> 16);
  }

  float BFloat16ToFloat(uint16_t value)
  {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
  }

  ScalarToken ParseScalarToken(const std::string &token)
  {
    const std::string lower = ToLower(token);
    if (lower == "nan")
    {
      return {ScalarToken::Kind::kNaN, 0.0L, false, false};
    }
    if (lower == "inf" || lower == "+inf" || lower == "infinity" || lower == "+infinity")
    {
      return {ScalarToken::Kind::kPosInf, 0.0L, false, false};
    }
    if (lower == "-inf" || lower == "-infinity")
    {
      return {ScalarToken::Kind::kNegInf, 0.0L, false, false};
    }
    if (lower == "true")
    {
      return {ScalarToken::Kind::kBool, 0.0L, true, false};
    }
    if (lower == "false")
    {
      return {ScalarToken::Kind::kBool, 0.0L, false, false};
    }

    errno = 0;
    char *end = nullptr;
    const long double value = std::strtold(token.c_str(), &end);
    CHECK_RET(end != token.c_str() && *end == '\0' && errno == 0,
              throw std::runtime_error("invalid numeric token: " + token));
    return {ScalarToken::Kind::kFinite, value, false, value == 0.0L && !token.empty() && token[0] == '-'};
  }

  template <typename T>
  void EncodeRealValue(std::vector<uint8_t> *bytes, const ScalarToken &token)
  {
    T value{};
    if constexpr (std::is_same_v<T, bool>)
    {
      value = token.kind == ScalarToken::Kind::kBool ? token.boolValue : (token.value != 0.0L);
    }
    else if constexpr (std::is_floating_point_v<T>)
    {
      if (token.kind == ScalarToken::Kind::kNaN)
      {
        value = std::numeric_limits<T>::quiet_NaN();
      }
      else if (token.kind == ScalarToken::Kind::kPosInf)
      {
        value = std::numeric_limits<T>::infinity();
      }
      else if (token.kind == ScalarToken::Kind::kNegInf)
      {
        value = -std::numeric_limits<T>::infinity();
      }
      else if (token.kind == ScalarToken::Kind::kBool)
      {
        value = token.boolValue ? static_cast<T>(1) : static_cast<T>(0);
      }
      else
      {
        value = static_cast<T>(token.value);
      }
    }
    else
    {
      if (token.kind == ScalarToken::Kind::kBool)
      {
        value = token.boolValue ? static_cast<T>(1) : static_cast<T>(0);
      }
      else
      {
        value = static_cast<T>(std::llround(token.value));
      }
    }
    AppendBytes(bytes, value);
  }

  std::vector<uint8_t> EncodeData(const std::vector<std::string> &tokens, DType dtype)
  {
    std::vector<uint8_t> bytes;
    bytes.reserve(tokens.size() * DTypeSize(dtype));
    for (const std::string &token : tokens)
    {
      switch (dtype)
      {
      case DType::kFloat:
      {
        const ScalarToken parsed = ParseScalarToken(token);
        EncodeRealValue<float>(&bytes, parsed);
        break;
      }
      case DType::kFloat16:
      {
        const ScalarToken parsed = ParseScalarToken(token);
        float value = 0.0f;
        if (parsed.kind == ScalarToken::Kind::kNaN)
        {
          value = std::numeric_limits<float>::quiet_NaN();
        }
        else if (parsed.kind == ScalarToken::Kind::kPosInf)
        {
          value = std::numeric_limits<float>::infinity();
        }
        else if (parsed.kind == ScalarToken::Kind::kNegInf)
        {
          value = -std::numeric_limits<float>::infinity();
        }
        else if (parsed.kind == ScalarToken::Kind::kBool)
        {
          value = parsed.boolValue ? 1.0f : 0.0f;
        }
        else
        {
          value = static_cast<float>(parsed.value);
        }
        AppendBytes(&bytes, aclFloatToFloat16(value));
        break;
      }
      case DType::kBFloat16:
      {
        const ScalarToken parsed = ParseScalarToken(token);
        float value = 0.0f;
        if (parsed.kind == ScalarToken::Kind::kNaN)
        {
          value = std::numeric_limits<float>::quiet_NaN();
        }
        else if (parsed.kind == ScalarToken::Kind::kPosInf)
        {
          value = std::numeric_limits<float>::infinity();
        }
        else if (parsed.kind == ScalarToken::Kind::kNegInf)
        {
          value = -std::numeric_limits<float>::infinity();
        }
        else if (parsed.kind == ScalarToken::Kind::kBool)
        {
          value = parsed.boolValue ? 1.0f : 0.0f;
        }
        else
        {
          value = static_cast<float>(parsed.value);
        }
        AppendBytes(&bytes, FloatToBFloat16(value));
        break;
      }
      case DType::kDouble:
      {
        const ScalarToken parsed = ParseScalarToken(token);
        EncodeRealValue<double>(&bytes, parsed);
        break;
      }
      case DType::kInt32:
      {
        const ScalarToken parsed = ParseScalarToken(token);
        EncodeRealValue<int32_t>(&bytes, parsed);
        break;
      }
      case DType::kInt64:
      {
        const ScalarToken parsed = ParseScalarToken(token);
        EncodeRealValue<int64_t>(&bytes, parsed);
        break;
      }
      case DType::kInt16:
      {
        const ScalarToken parsed = ParseScalarToken(token);
        EncodeRealValue<int16_t>(&bytes, parsed);
        break;
      }
      case DType::kInt8:
      {
        const ScalarToken parsed = ParseScalarToken(token);
        EncodeRealValue<int8_t>(&bytes, parsed);
        break;
      }
      case DType::kUInt8:
      {
        const ScalarToken parsed = ParseScalarToken(token);
        EncodeRealValue<uint8_t>(&bytes, parsed);
        break;
      }
      case DType::kBool:
      {
        const ScalarToken parsed = ParseScalarToken(token);
        EncodeRealValue<bool>(&bytes, parsed);
        break;
      }
      case DType::kComplex32:
      {
        const auto complexPair = ParseComplexToken(token);
        const float realValue = static_cast<float>(complexPair.first.kind == ScalarToken::Kind::kBool
                                                       ? (complexPair.first.boolValue ? 1.0L : 0.0L)
                                                       : complexPair.first.value);
        const float imagValue = static_cast<float>(complexPair.second.kind == ScalarToken::Kind::kBool
                                                       ? (complexPair.second.boolValue ? 1.0L : 0.0L)
                                                       : complexPair.second.value);
        AppendBytes(&bytes, aclFloatToFloat16(realValue));
        AppendBytes(&bytes, aclFloatToFloat16(imagValue));
        break;
      }
      case DType::kComplex64:
      {
        const auto complexPair = ParseComplexToken(token);
        const float realValue = static_cast<float>(complexPair.first.kind == ScalarToken::Kind::kBool
                                                       ? (complexPair.first.boolValue ? 1.0L : 0.0L)
                                                       : complexPair.first.value);
        const float imagValue = static_cast<float>(complexPair.second.kind == ScalarToken::Kind::kBool
                                                       ? (complexPair.second.boolValue ? 1.0L : 0.0L)
                                                       : complexPair.second.value);
        AppendBytes(&bytes, realValue);
        AppendBytes(&bytes, imagValue);
        break;
      }
      }
    }
    return bytes;
  }

  std::string FormatFloatString(long double value, bool keepNegativeZero)
  {
    if (std::isnan(static_cast<double>(value)))
    {
      return "nan";
    }
    if (std::isinf(static_cast<double>(value)))
    {
      return value > 0 ? "inf" : "-inf";
    }
    if (value == 0.0L && keepNegativeZero)
    {
      return "-0.0";
    }
    std::ostringstream oss;
    oss << std::setprecision(17) << static_cast<double>(value);
    return oss.str();
  }

  std::vector<std::string> DecodeData(const std::vector<uint8_t> &bytes, DType dtype)
  {
    std::vector<std::string> values;
    const size_t elemSize = DTypeSize(dtype);
    if (elemSize == 0)
    {
      return values;
    }
    values.reserve(bytes.size() / elemSize);
    for (size_t offset = 0; offset + elemSize <= bytes.size(); offset += elemSize)
    {
      switch (dtype)
      {
      case DType::kFloat:
      {
        float value = 0.0f;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(FormatFloatString(value, std::signbit(value) && value == 0.0f));
        break;
      }
      case DType::kFloat16:
      {
        aclFloat16 value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        const float decoded = aclFloat16ToFloat(value);
        values.push_back(FormatFloatString(decoded, std::signbit(decoded) && decoded == 0.0f));
        break;
      }
      case DType::kBFloat16:
      {
        uint16_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        const float decoded = BFloat16ToFloat(value);
        values.push_back(FormatFloatString(decoded, std::signbit(decoded) && decoded == 0.0f));
        break;
      }
      case DType::kDouble:
      {
        double value = 0.0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(FormatFloatString(value, std::signbit(value) && value == 0.0));
        break;
      }
      case DType::kInt32:
      {
        int32_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(std::to_string(value));
        break;
      }
      case DType::kInt64:
      {
        int64_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(std::to_string(value));
        break;
      }
      case DType::kInt16:
      {
        int16_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(std::to_string(value));
        break;
      }
      case DType::kInt8:
      {
        int8_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(std::to_string(static_cast<int>(value)));
        break;
      }
      case DType::kUInt8:
      {
        uint8_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(std::to_string(static_cast<unsigned int>(value)));
        break;
      }
      case DType::kBool:
      {
        bool value = false;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        values.push_back(value ? "true" : "false");
        break;
      }
      case DType::kComplex32:
      {
        aclFloat16 realHalf = 0;
        aclFloat16 imagHalf = 0;
        std::memcpy(&realHalf, bytes.data() + offset, sizeof(realHalf));
        std::memcpy(&imagHalf, bytes.data() + offset + sizeof(realHalf), sizeof(imagHalf));
        const float realValue = aclFloat16ToFloat(realHalf);
        const float imagValue = aclFloat16ToFloat(imagHalf);
        values.push_back(FormatFloatString(realValue, std::signbit(realValue) && realValue == 0.0f) + "," +
                         FormatFloatString(imagValue, std::signbit(imagValue) && imagValue == 0.0f));
        break;
      }
      case DType::kComplex64:
      {
        float realValue = 0.0f;
        float imagValue = 0.0f;
        std::memcpy(&realValue, bytes.data() + offset, sizeof(realValue));
        std::memcpy(&imagValue, bytes.data() + offset + sizeof(realValue), sizeof(imagValue));
        values.push_back(FormatFloatString(realValue, std::signbit(realValue) && realValue == 0.0f) + "," +
                         FormatFloatString(imagValue, std::signbit(imagValue) && imagValue == 0.0f));
        break;
      }
      }
    }
    return values;
  }

  std::vector<std::string> MakeDefaultData(size_t count, DType dtype)
  {
    if (dtype == DType::kBool)
    {
      return std::vector<std::string>(count, "false");
    }
    return std::vector<std::string>(count, "0");
  }

  void NormalizeCase(CaseSpec *spec)
  {
    const bool lhsScalar = LhsIsScalar(spec->api);
    const bool rhsScalar = RhsIsScalar(spec->api);

    if (!lhsScalar && spec->lhsShape.empty())
    {
      throw std::runtime_error("tensor lhs requires lhs_shape");
    }
    if (!rhsScalar && spec->rhsShape.empty())
    {
      throw std::runtime_error("tensor rhs requires rhs_shape");
    }
    if (lhsScalar && spec->lhsData.empty())
    {
      spec->lhsData = MakeDefaultData(1, spec->lhsDType);
    }
    if (!lhsScalar && spec->lhsData.empty())
    {
      spec->lhsData = MakeDefaultData(static_cast<size_t>(GetShapeSize(spec->lhsShape)), spec->lhsDType);
    }
    if (rhsScalar && spec->rhsData.empty())
    {
      spec->rhsData = MakeDefaultData(1, spec->rhsDType);
    }
    if (!rhsScalar && spec->rhsData.empty())
    {
      spec->rhsData = MakeDefaultData(static_cast<size_t>(GetShapeSize(spec->rhsShape)), spec->rhsDType);
    }
    if (spec->alphaData.empty())
    {
      spec->alphaData = MakeDefaultData(1, spec->alphaDType);
    }

    const size_t lhsSize = lhsScalar ? 1U : static_cast<size_t>(GetShapeSize(spec->lhsShape));
    const size_t rhsSize = rhsScalar ? 1U : static_cast<size_t>(GetShapeSize(spec->rhsShape));
    if (spec->lhsData.size() != lhsSize)
    {
      throw std::runtime_error("lhs_data size mismatch in " + spec->caseId);
    }
    if (spec->rhsData.size() != rhsSize)
    {
      throw std::runtime_error("rhs_data size mismatch in " + spec->caseId);
    }
    if (spec->alphaData.size() != 1)
    {
      throw std::runtime_error("alpha_data size mismatch in " + spec->caseId);
    }

    if (spec->outShape.empty())
    {
      if (spec->api == ApiKind::kAdd || spec->api == ApiKind::kInplaceAdd)
      {
        spec->outShape = IsInplace(spec->api) ? spec->lhsShape : InferBroadcastShape(spec->lhsShape, spec->rhsShape);
      }
      else if (spec->api == ApiKind::kAdds || spec->api == ApiKind::kInplaceAdds)
      {
        spec->outShape = spec->lhsShape;
      }
      else
      {
        spec->outShape = spec->rhsShape;
      }
    }

    if (spec->tolerance.atol == 0.0 && spec->tolerance.rtol == 0.0)
    {
      spec->tolerance = DefaultTolerance(spec->outDType);
    }
  }

  CaseResult MakeBaseResult(const CaseSpec &spec, Phase phase)
  {
    CaseResult result;
    result.caseId = spec.caseId;
    result.title = spec.title;
    result.phase = phase == Phase::kApi ? "api" : "exec";
    result.api = ApiKindToString(spec.api);
    return result;
  }

  bool CurrentPlatformSupportsBFloat16(const std::string &socName)
  {
    const std::string lower = ToLower(socName);
    return lower.find("950") != std::string::npos || lower.find("910b") != std::string::npos ||
           lower.find("910_93") != std::string::npos || lower.find("a3") != std::string::npos;
  }

  bool CaseTouchesBFloat16(const CaseSpec &spec)
  {
    return spec.lhsDType == DType::kBFloat16 || spec.rhsDType == DType::kBFloat16 || spec.alphaDType == DType::kBFloat16 ||
           spec.outDType == DType::kBFloat16;
  }

  bool ShouldSkipCase(const CaseSpec &spec, const RuntimeContext &runtime, std::string *message)
  {
    if (CaseTouchesBFloat16(spec) && !CurrentPlatformSupportsBFloat16(runtime.socName))
    {
      *message = "current platform does not advertise bfloat16 support";
      return true;
    }
    if (spec.lhsDType == DType::kDouble || spec.rhsDType == DType::kDouble || spec.outDType == DType::kDouble)
    {
      *message = "ACL_DOUBLE may route to AICPU on simulator and is intentionally avoided in embedded exec cases";
      return spec.allowSkip;
    }
    return false;
  }

  int InitRuntime(RuntimeContext *runtime)
  {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(runtime->deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(&runtime->stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    const char *socName = aclrtGetSocName();
    runtime->socName = socName == nullptr ? "unknown" : socName;
    return ACL_SUCCESS;
  }

  void FinalizeRuntime(RuntimeContext *runtime)
  {
    if (runtime->stream != nullptr)
    {
      aclrtDestroyStream(runtime->stream);
      runtime->stream = nullptr;
    }
    aclrtResetDevice(runtime->deviceId);
    aclFinalize();
  }

  int CreateAclTensor(const std::vector<uint8_t> &hostData, const std::vector<int64_t> &shape, DType dtype,
                      DeviceTensor *deviceTensor)
  {
    const size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * DTypeSize(dtype);
    const size_t allocBytes = bytes == 0 ? 1 : bytes;
    auto ret = aclrtMalloc(&deviceTensor->deviceAddr, allocBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    if (bytes > 0)
    {
      ret = aclrtMemcpy(deviceTensor->deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy host->device failed. ERROR: %d\n", ret); return ret);
    }
    std::vector<int64_t> strides = ComputeStrides(shape);
    deviceTensor->tensor = aclCreateTensor(shape.data(), shape.size(), ToAclDataType(dtype), strides.data(), 0,
                                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
                                           deviceTensor->deviceAddr);
    CHECK_RET(deviceTensor->tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
  }

  int CreateAclScalar(const std::vector<std::string> &tokens, DType dtype, HostScalar *hostScalar)
  {
    CHECK_RET(tokens.size() == 1, LOG_PRINT("scalar token size must be 1.\n"); return ACL_ERROR_FAILURE);
    hostScalar->storage = EncodeData(tokens, dtype);
    hostScalar->scalar = aclCreateScalar(hostScalar->storage.data(), ToAclDataType(dtype));
    CHECK_RET(hostScalar->scalar != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
  }

  void DestroyTensor(DeviceTensor *deviceTensor)
  {
    if (deviceTensor->tensor != nullptr)
    {
      aclDestroyTensor(deviceTensor->tensor);
      deviceTensor->tensor = nullptr;
    }
    if (deviceTensor->deviceAddr != nullptr)
    {
      aclrtFree(deviceTensor->deviceAddr);
      deviceTensor->deviceAddr = nullptr;
    }
  }

  void DestroyScalar(HostScalar *hostScalar)
  {
    if (hostScalar->scalar != nullptr)
    {
      aclDestroyScalar(hostScalar->scalar);
      hostScalar->scalar = nullptr;
    }
    hostScalar->storage.clear();
  }

  bool ReadbackTensor(const DeviceTensor &deviceTensor, const std::vector<int64_t> &shape, DType dtype,
                      std::vector<std::string> *tokens)
  {
    const size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * DTypeSize(dtype);
    std::vector<uint8_t> host(bytes == 0 ? 1 : bytes, 0);
    if (bytes > 0)
    {
      const auto ret = aclrtMemcpy(host.data(), bytes, deviceTensor.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, return false);
    }
    *tokens = DecodeData(host, dtype);
    return true;
  }

  std::vector<int64_t> UnravelIndex(int64_t linearIndex, const std::vector<int64_t> &shape)
  {
    if (shape.empty())
    {
      return {};
    }
    std::vector<int64_t> index(shape.size(), 0);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i)
    {
      const int64_t dim = shape[static_cast<size_t>(i)];
      if (dim == 0)
      {
        index[static_cast<size_t>(i)] = 0;
        continue;
      }
      index[static_cast<size_t>(i)] = linearIndex % dim;
      linearIndex /= dim;
    }
    return index;
  }

  int64_t BroadcastOffset(const std::vector<int64_t> &outIndex, const std::vector<int64_t> &shape)
  {
    if (shape.empty())
    {
      return 0;
    }
    const std::vector<int64_t> strides = ComputeStrides(shape);
    const size_t rankGap = outIndex.size() - shape.size();
    int64_t offset = 0;
    for (size_t i = 0; i < shape.size(); ++i)
    {
      const int64_t dim = shape[i];
      const int64_t idx = dim == 1 ? 0 : outIndex[rankGap + i];
      offset += idx * strides[i];
    }
    return offset;
  }

  RefValue TokenToRefValue(const std::string &token)
  {
    if (token.find(',') != std::string::npos)
    {
      const auto complexPair = ParseComplexToken(token);
      RefValue value;
      value.isComplex = true;
      value.complexValue = {
          complexPair.first.kind == ScalarToken::Kind::kBool ? (complexPair.first.boolValue ? 1.0L : 0.0L)
                                                             : complexPair.first.value,
          complexPair.second.kind == ScalarToken::Kind::kBool ? (complexPair.second.boolValue ? 1.0L : 0.0L)
                                                              : complexPair.second.value};
      return value;
    }
    const ScalarToken parsed = ParseScalarToken(token);
    return {parsed.kind, parsed.value, parsed.boolValue, parsed.negativeZero};
  }

  RefReal RefValueToNumber(const RefValue &value)
  {
    CHECK_RET(!value.isComplex, throw std::runtime_error("RefValueToNumber does not accept complex values"));
    switch (value.kind)
    {
    case ScalarToken::Kind::kBool:
      return value.boolValue ? 1.0L : 0.0L;
    case ScalarToken::Kind::kPosInf:
      return std::numeric_limits<long double>::infinity();
    case ScalarToken::Kind::kNegInf:
      return -std::numeric_limits<long double>::infinity();
    case ScalarToken::Kind::kNaN:
      return std::numeric_limits<long double>::quiet_NaN();
    case ScalarToken::Kind::kFinite:
      return value.value;
    }
    return value.value;
  }

  bool IsAddsBoolSpecialCastCase(const CaseSpec &spec, const ScalarToken &rhsToken, const ScalarToken &alphaToken)
  {
    return spec.api == ApiKind::kAdds && spec.lhsDType == DType::kBool && spec.rhsDType == DType::kBool &&
           spec.alphaDType == DType::kBool && spec.outDType != DType::kBool &&
           rhsToken.kind == ScalarToken::Kind::kBool && rhsToken.boolValue &&
           alphaToken.kind == ScalarToken::Kind::kBool && alphaToken.boolValue;
  }

  RefReal MulAddRefReal(RefReal lhs, RefReal alpha, RefReal rhs)
  {
    if (std::isfinite(lhs) && std::isfinite(alpha) && std::isfinite(rhs))
    {
      return std::fma(alpha, rhs, lhs);
    }
    return lhs + alpha * rhs;
  }

  RefValue AddRef(const RefValue &lhs, const RefValue &rhs, const RefValue &alpha)
  {
    if (lhs.isComplex || rhs.isComplex)
    {
      RefValue result;
      result.isComplex = true;
      const RefComplex lhsComplex = lhs.isComplex ? lhs.complexValue : RefComplex(RefValueToNumber(lhs), 0.0L);
      const RefComplex rhsComplex = rhs.isComplex ? rhs.complexValue : RefComplex(RefValueToNumber(rhs), 0.0L);
      result.complexValue = lhsComplex + RefValueToNumber(alpha) * rhsComplex;
      return result;
    }
    const RefReal result = MulAddRefReal(RefValueToNumber(lhs), RefValueToNumber(alpha), RefValueToNumber(rhs));
    return {std::isnan(static_cast<double>(result))   ? ScalarToken::Kind::kNaN
            : std::isinf(static_cast<double>(result)) ? (result > 0 ? ScalarToken::Kind::kPosInf
                                                                    : ScalarToken::Kind::kNegInf)
                                                      : ScalarToken::Kind::kFinite,
            result,
            false,
            std::signbit(static_cast<double>(result)) && result == 0.0L};
  }

  std::vector<RefValue> ComputeExpectedValues(const CaseSpec &spec)
  {
    const int64_t totalSize = GetShapeSize(spec.outShape);
    std::vector<RefValue> values;
    values.reserve(static_cast<size_t>(totalSize));
    const ScalarToken alphaToken = ParseScalarToken(spec.alphaData[0]);
    const RefValue alphaValue{alphaToken.kind, alphaToken.value, alphaToken.boolValue, alphaToken.negativeZero};

    for (int64_t i = 0; i < totalSize; ++i)
    {
      const std::vector<int64_t> outIndex = UnravelIndex(i, spec.outShape);
      RefValue lhsValue;
      RefValue rhsValue;
      if (LhsIsScalar(spec.api))
      {
        lhsValue = TokenToRefValue(spec.lhsData[0]);
      }
      else
      {
        lhsValue = TokenToRefValue(spec.lhsData[static_cast<size_t>(BroadcastOffset(outIndex, spec.lhsShape))]);
      }
      if (RhsIsScalar(spec.api))
      {
        rhsValue = TokenToRefValue(spec.rhsData[0]);
      }
      else
      {
        rhsValue = TokenToRefValue(spec.rhsData[static_cast<size_t>(BroadcastOffset(outIndex, spec.rhsShape))]);
      }
      RefValue ref = AddRef(lhsValue, rhsValue, alphaValue);
      if (RhsIsScalar(spec.api) && IsAddsBoolSpecialCastCase(spec, ParseScalarToken(spec.rhsData[0]), alphaToken))
      {
        ref = {ScalarToken::Kind::kBool, 0.0L, RefValueToNumber(ref) != 0.0L, false};
      }
      values.push_back(ref);
    }
    return values;
  }

  std::vector<std::string> EncodeExpectedValues(const std::vector<RefValue> &values, DType dtype)
  {
    std::vector<uint8_t> bytes;
    bytes.reserve(values.size() * DTypeSize(dtype));
    for (const RefValue &value : values)
    {
      switch (dtype)
      {
      case DType::kFloat:
      {
        float out = 0.0f;
        if (value.kind == ScalarToken::Kind::kNaN)
        {
          out = std::numeric_limits<float>::quiet_NaN();
        }
        else if (value.kind == ScalarToken::Kind::kPosInf)
        {
          out = std::numeric_limits<float>::infinity();
        }
        else if (value.kind == ScalarToken::Kind::kNegInf)
        {
          out = -std::numeric_limits<float>::infinity();
        }
        else
        {
          out = static_cast<float>(RefValueToNumber(value));
        }
        AppendBytes(&bytes, out);
        break;
      }
      case DType::kFloat16:
      {
        float out = 0.0f;
        if (value.kind == ScalarToken::Kind::kNaN)
        {
          out = std::numeric_limits<float>::quiet_NaN();
        }
        else if (value.kind == ScalarToken::Kind::kPosInf)
        {
          out = std::numeric_limits<float>::infinity();
        }
        else if (value.kind == ScalarToken::Kind::kNegInf)
        {
          out = -std::numeric_limits<float>::infinity();
        }
        else
        {
          out = static_cast<float>(RefValueToNumber(value));
        }
        AppendBytes(&bytes, aclFloatToFloat16(out));
        break;
      }
      case DType::kBFloat16:
      {
        float out = 0.0f;
        if (value.kind == ScalarToken::Kind::kNaN)
        {
          out = std::numeric_limits<float>::quiet_NaN();
        }
        else if (value.kind == ScalarToken::Kind::kPosInf)
        {
          out = std::numeric_limits<float>::infinity();
        }
        else if (value.kind == ScalarToken::Kind::kNegInf)
        {
          out = -std::numeric_limits<float>::infinity();
        }
        else
        {
          out = static_cast<float>(RefValueToNumber(value));
        }
        AppendBytes(&bytes, FloatToBFloat16(out));
        break;
      }
      case DType::kDouble:
      {
        double out = 0.0;
        if (value.kind == ScalarToken::Kind::kNaN)
        {
          out = std::numeric_limits<double>::quiet_NaN();
        }
        else if (value.kind == ScalarToken::Kind::kPosInf)
        {
          out = std::numeric_limits<double>::infinity();
        }
        else if (value.kind == ScalarToken::Kind::kNegInf)
        {
          out = -std::numeric_limits<double>::infinity();
        }
        else
        {
          out = static_cast<double>(RefValueToNumber(value));
        }
        AppendBytes(&bytes, out);
        break;
      }
      case DType::kInt32:
        AppendBytes(&bytes, static_cast<int32_t>(RefValueToNumber(value)));
        break;
      case DType::kInt64:
        AppendBytes(&bytes, static_cast<int64_t>(RefValueToNumber(value)));
        break;
      case DType::kInt16:
        AppendBytes(&bytes, static_cast<int16_t>(RefValueToNumber(value)));
        break;
      case DType::kInt8:
        AppendBytes(&bytes, static_cast<int8_t>(RefValueToNumber(value)));
        break;
      case DType::kUInt8:
        AppendBytes(&bytes, static_cast<uint8_t>(RefValueToNumber(value)));
        break;
      case DType::kBool:
        AppendBytes(&bytes, RefValueToNumber(value) != 0.0L);
        break;
      case DType::kComplex32:
      {
        const float realValue = static_cast<float>(value.isComplex ? value.complexValue.real() : RefValueToNumber(value));
        const float imagValue = static_cast<float>(value.isComplex ? value.complexValue.imag() : 0.0L);
        AppendBytes(&bytes, aclFloatToFloat16(realValue));
        AppendBytes(&bytes, aclFloatToFloat16(imagValue));
        break;
      }
      case DType::kComplex64:
      {
        const float realValue = static_cast<float>(value.isComplex ? value.complexValue.real() : RefValueToNumber(value));
        const float imagValue = static_cast<float>(value.isComplex ? value.complexValue.imag() : 0.0L);
        AppendBytes(&bytes, realValue);
        AppendBytes(&bytes, imagValue);
        break;
      }
      }
    }
    return DecodeData(bytes, dtype);
  }

  std::vector<std::string> ComputeExpectedData(const CaseSpec &spec)
  {
    return EncodeExpectedValues(ComputeExpectedValues(spec), spec.outDType);
  }

  bool TokenEquals(const std::string &actual, const std::string &expect, DType dtype, const Tolerance &tolerance,
                   double *absError, double *relError)
  {
    *absError = 0.0;
    *relError = 0.0;
    if (IsComplexType(dtype))
    {
      const auto actualPair = ParseComplexToken(actual);
      const auto expectPair = ParseComplexToken(expect);
      const double actualReal = static_cast<double>(actualPair.first.kind == ScalarToken::Kind::kBool
                                                        ? (actualPair.first.boolValue ? 1.0L : 0.0L)
                                                        : actualPair.first.value);
      const double actualImag = static_cast<double>(actualPair.second.kind == ScalarToken::Kind::kBool
                                                        ? (actualPair.second.boolValue ? 1.0L : 0.0L)
                                                        : actualPair.second.value);
      const double expectReal = static_cast<double>(expectPair.first.kind == ScalarToken::Kind::kBool
                                                        ? (expectPair.first.boolValue ? 1.0L : 0.0L)
                                                        : expectPair.first.value);
      const double expectImag = static_cast<double>(expectPair.second.kind == ScalarToken::Kind::kBool
                                                        ? (expectPair.second.boolValue ? 1.0L : 0.0L)
                                                        : expectPair.second.value);
      const double absReal = std::fabs(actualReal - expectReal);
      const double absImag = std::fabs(actualImag - expectImag);
      const double relReal = std::fabs(expectReal) > 0.0 ? (absReal / std::fabs(expectReal)) : absReal;
      const double relImag = std::fabs(expectImag) > 0.0 ? (absImag / std::fabs(expectImag)) : absImag;
      *absError = std::max(absReal, absImag);
      *relError = std::max(relReal, relImag);
      return absReal <= tolerance.atol + tolerance.rtol * std::fabs(expectReal) &&
             absImag <= tolerance.atol + tolerance.rtol * std::fabs(expectImag);
    }
    if (dtype == DType::kBool)
    {
      return ToLower(actual) == ToLower(expect);
    }
    if (!IsFloatingType(dtype))
    {
      return actual == expect;
    }

    const ScalarToken actualToken = ParseScalarToken(actual);
    const ScalarToken expectToken = ParseScalarToken(expect);
    if (expectToken.kind == ScalarToken::Kind::kNaN)
    {
      return actualToken.kind == ScalarToken::Kind::kNaN;
    }
    if (expectToken.kind == ScalarToken::Kind::kPosInf || expectToken.kind == ScalarToken::Kind::kNegInf)
    {
      return actualToken.kind == expectToken.kind;
    }

    const double actualValue = static_cast<double>(actualToken.value);
    const double expectValue = static_cast<double>(expectToken.value);
    *absError = std::fabs(actualValue - expectValue);
    *relError = std::fabs(expectValue) > 0.0 ? (*absError / std::fabs(expectValue)) : *absError;
    return *absError <= tolerance.atol + tolerance.rtol * std::fabs(expectValue);
  }

  bool CompareExecResult(const CaseSpec &spec, const std::vector<std::string> &expectedData,
                         const std::vector<std::string> &actualData, CaseResult *result)
  {
    result->expected = expectedData;
    result->actual = actualData;
    result->pass = true;
    for (size_t i = 0; i < actualData.size(); ++i)
    {
      double absError = 0.0;
      double relError = 0.0;
      if (!TokenEquals(actualData[i], expectedData[i], spec.outDType, spec.tolerance, &absError, &relError))
      {
        result->pass = false;
        result->firstMismatchIndex = static_cast<int64_t>(i);
        result->maxAbsError = absError;
        result->maxRelError = relError;
        result->message = "value mismatch";
        return false;
      }
      result->maxAbsError = std::max(result->maxAbsError, absError);
      result->maxRelError = std::max(result->maxRelError, relError);
    }
    result->message = "ok";
    return true;
  }

  CaseResult RunApiCaseCore(const CaseSpec &spec, const RuntimeContext &runtime)
  {
    CaseResult result = MakeBaseResult(spec, Phase::kApi);
    std::string skipMessage;
    if (ShouldSkipCase(spec, runtime, &skipMessage))
    {
      result.pass = spec.allowSkip;
      result.skipped = spec.allowSkip;
      result.message = spec.skipReason.empty() ? skipMessage : spec.skipReason;
      result.statusCode = spec.expectStatus;
      return result;
    }

    DeviceTensor lhsTensor;
    DeviceTensor rhsTensor;
    DeviceTensor outTensor;
    HostScalar lhsScalar;
    HostScalar rhsScalar;
    HostScalar alphaScalar;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto finish = [&]()
    {
      DestroyTensor(&lhsTensor);
      DestroyTensor(&rhsTensor);
      DestroyTensor(&outTensor);
      DestroyScalar(&lhsScalar);
      DestroyScalar(&rhsScalar);
      DestroyScalar(&alphaScalar);
      return result;
    };

    try
    {
      if (LhsIsScalar(spec.api))
      {
        auto ret = CreateAclScalar(spec.lhsData, spec.lhsDType, &lhsScalar);
        CHECK_RET(ret == ACL_SUCCESS, result.statusCode = ret; result.message = "failed to create lhs scalar"; return finish());
      }
      else
      {
        auto ret = CreateAclTensor(EncodeData(spec.lhsData, spec.lhsDType), spec.lhsShape, spec.lhsDType, &lhsTensor);
        CHECK_RET(ret == ACL_SUCCESS, result.statusCode = ret; result.message = "failed to create lhs tensor"; return finish());
      }

      if (RhsIsScalar(spec.api))
      {
        auto ret = CreateAclScalar(spec.rhsData, spec.rhsDType, &rhsScalar);
        CHECK_RET(ret == ACL_SUCCESS, result.statusCode = ret; result.message = "failed to create rhs scalar"; return finish());
      }
      else
      {
        auto ret = CreateAclTensor(EncodeData(spec.rhsData, spec.rhsDType), spec.rhsShape, spec.rhsDType, &rhsTensor);
        CHECK_RET(ret == ACL_SUCCESS, result.statusCode = ret; result.message = "failed to create rhs tensor"; return finish());
      }

      {
        auto ret = CreateAclScalar(spec.alphaData, spec.alphaDType, &alphaScalar);
        CHECK_RET(ret == ACL_SUCCESS, result.statusCode = ret; result.message = "failed to create alpha scalar"; return finish());
      }

      if (!IsInplace(spec.api))
      {
        const std::vector<uint8_t> outBytes(static_cast<size_t>(GetShapeSize(spec.outShape)) * DTypeSize(spec.outDType), 0);
        auto ret = CreateAclTensor(outBytes, spec.outShape, spec.outDType, &outTensor);
        CHECK_RET(ret == ACL_SUCCESS, result.statusCode = ret; result.message = "failed to create out tensor"; return finish());
      }

      aclTensor *lhsTensorArg = spec.nullWhich == "lhs" ? nullptr : lhsTensor.tensor;
      const aclTensor *rhsTensorArg = spec.nullWhich == "rhs" ? nullptr : rhsTensor.tensor;
      const aclScalar *lhsScalarArg = spec.nullWhich == "lhs" ? nullptr : lhsScalar.scalar;
      const aclScalar *rhsScalarArg = spec.nullWhich == "rhs" ? nullptr : rhsScalar.scalar;
      const aclScalar *alphaArg = spec.nullWhich == "alpha" ? nullptr : alphaScalar.scalar;
      aclTensor *outArg = spec.nullWhich == "out" ? nullptr : outTensor.tensor;

      switch (spec.api)
      {
      case ApiKind::kAdd:
        result.statusCode = aclnnAddGetWorkspaceSize(lhsTensorArg, rhsTensorArg, alphaArg, outArg, &workspaceSize, &executor);
        break;
      case ApiKind::kAdds:
        result.statusCode = aclnnAddsGetWorkspaceSize(lhsTensorArg, rhsScalarArg, alphaArg, outArg, &workspaceSize, &executor);
        break;
      case ApiKind::kInplaceAdd:
        result.statusCode = aclnnInplaceAddGetWorkspaceSize(lhsTensorArg, rhsTensorArg, alphaArg, &workspaceSize, &executor);
        break;
      case ApiKind::kInplaceAdds:
        result.statusCode = aclnnInplaceAddsGetWorkspaceSize(lhsTensorArg, rhsScalarArg, alphaArg, &workspaceSize, &executor);
        break;
      case ApiKind::kAddV3:
        result.statusCode = aclnnAddV3GetWorkspaceSize(lhsScalarArg, rhsTensorArg, alphaArg, outArg, &workspaceSize, &executor);
        break;
      case ApiKind::kInplaceAddV3:
        result.statusCode = aclnnInplaceAddV3GetWorkspaceSize(lhsScalarArg, rhsTensorArg, alphaArg, &workspaceSize, &executor);
        break;
      }
      result.pass = (result.statusCode == spec.expectStatus);
      result.message = result.pass ? "ok" : "unexpected status code";
    }
    catch (const std::exception &ex)
    {
      result.statusCode = ACL_ERROR_FAILURE;
      result.pass = false;
      result.message = ex.what();
    }
    return finish();
  }

  CaseResult RunExecCaseCore(const CaseSpec &spec, const RuntimeContext &runtime)
  {
    CaseResult result = MakeBaseResult(spec, Phase::kExec);
    std::string skipMessage;
    if (ShouldSkipCase(spec, runtime, &skipMessage))
    {
      result.pass = spec.allowSkip;
      result.skipped = spec.allowSkip;
      result.message = spec.skipReason.empty() ? skipMessage : spec.skipReason;
      result.statusCode = spec.expectStatus;
      return result;
    }

    DeviceTensor lhsTensor;
    DeviceTensor rhsTensor;
    DeviceTensor outTensor;
    HostScalar lhsScalar;
    HostScalar rhsScalar;
    HostScalar alphaScalar;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;

    auto finish = [&]()
    {
      DestroyTensor(&lhsTensor);
      DestroyTensor(&rhsTensor);
      DestroyTensor(&outTensor);
      DestroyScalar(&lhsScalar);
      DestroyScalar(&rhsScalar);
      DestroyScalar(&alphaScalar);
      if (workspaceAddr != nullptr)
      {
        aclrtFree(workspaceAddr);
      }
      return result;
    };

    try
    {
      const std::vector<std::string> expectedData = ComputeExpectedData(spec);
      result.expected = expectedData;

      if (LhsIsScalar(spec.api))
      {
        auto ret = CreateAclScalar(spec.lhsData, spec.lhsDType, &lhsScalar);
        CHECK_RET(ret == ACL_SUCCESS, result.statusCode = ret; result.message = "failed to create lhs scalar"; return finish());
      }
      else
      {
        auto ret = CreateAclTensor(EncodeData(spec.lhsData, spec.lhsDType), spec.lhsShape, spec.lhsDType, &lhsTensor);
        CHECK_RET(ret == ACL_SUCCESS, result.statusCode = ret; result.message = "failed to create lhs tensor"; return finish());
      }

      if (RhsIsScalar(spec.api))
      {
        auto ret = CreateAclScalar(spec.rhsData, spec.rhsDType, &rhsScalar);
        CHECK_RET(ret == ACL_SUCCESS, result.statusCode = ret; result.message = "failed to create rhs scalar"; return finish());
      }
      else
      {
        auto ret = CreateAclTensor(EncodeData(spec.rhsData, spec.rhsDType), spec.rhsShape, spec.rhsDType, &rhsTensor);
        CHECK_RET(ret == ACL_SUCCESS, result.statusCode = ret; result.message = "failed to create rhs tensor"; return finish());
      }

      {
        auto ret = CreateAclScalar(spec.alphaData, spec.alphaDType, &alphaScalar);
        CHECK_RET(ret == ACL_SUCCESS, result.statusCode = ret; result.message = "failed to create alpha scalar"; return finish());
      }

      if (!IsInplace(spec.api))
      {
        const std::vector<uint8_t> outBytes(static_cast<size_t>(GetShapeSize(spec.outShape)) * DTypeSize(spec.outDType), 0);
        auto ret = CreateAclTensor(outBytes, spec.outShape, spec.outDType, &outTensor);
        CHECK_RET(ret == ACL_SUCCESS, result.statusCode = ret; result.message = "failed to create out tensor"; return finish());
      }

      switch (spec.api)
      {
      case ApiKind::kAdd:
        result.statusCode = aclnnAddGetWorkspaceSize(lhsTensor.tensor, rhsTensor.tensor, alphaScalar.scalar, outTensor.tensor,
                                                     &workspaceSize, &executor);
        break;
      case ApiKind::kAdds:
        result.statusCode = aclnnAddsGetWorkspaceSize(lhsTensor.tensor, rhsScalar.scalar, alphaScalar.scalar, outTensor.tensor,
                                                      &workspaceSize, &executor);
        break;
      case ApiKind::kInplaceAdd:
        result.statusCode = aclnnInplaceAddGetWorkspaceSize(lhsTensor.tensor, rhsTensor.tensor, alphaScalar.scalar, &workspaceSize,
                                                            &executor);
        break;
      case ApiKind::kInplaceAdds:
        result.statusCode = aclnnInplaceAddsGetWorkspaceSize(lhsTensor.tensor, rhsScalar.scalar, alphaScalar.scalar, &workspaceSize,
                                                             &executor);
        break;
      case ApiKind::kAddV3:
        result.statusCode = aclnnAddV3GetWorkspaceSize(lhsScalar.scalar, rhsTensor.tensor, alphaScalar.scalar, outTensor.tensor,
                                                       &workspaceSize, &executor);
        break;
      case ApiKind::kInplaceAddV3:
        result.statusCode = aclnnInplaceAddV3GetWorkspaceSize(lhsScalar.scalar, rhsTensor.tensor, alphaScalar.scalar,
                                                              &workspaceSize, &executor);
        break;
      }

      if (result.statusCode != spec.expectStatus)
      {
        result.pass = false;
        result.message = "unexpected status code from GetWorkspaceSize";
        return finish();
      }
      if (result.statusCode != ACLNN_SUCCESS)
      {
        result.pass = true;
        result.message = "expected non-success status";
        return finish();
      }

      if (workspaceSize > 0)
      {
        auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, result.statusCode = ret; result.message = "allocate workspace failed"; return finish());
      }

      switch (spec.api)
      {
      case ApiKind::kAdd:
        result.statusCode = aclnnAdd(workspaceAddr, workspaceSize, executor, runtime.stream);
        break;
      case ApiKind::kAdds:
        result.statusCode = aclnnAdds(workspaceAddr, workspaceSize, executor, runtime.stream);
        break;
      case ApiKind::kInplaceAdd:
        result.statusCode = aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, runtime.stream);
        break;
      case ApiKind::kInplaceAdds:
        result.statusCode = aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, runtime.stream);
        break;
      case ApiKind::kAddV3:
        result.statusCode = aclnnAddV3(workspaceAddr, workspaceSize, executor, runtime.stream);
        break;
      case ApiKind::kInplaceAddV3:
        result.statusCode = aclnnInplaceAddV3(workspaceAddr, workspaceSize, executor, runtime.stream);
        break;
      }
      CHECK_RET(result.statusCode == ACLNN_SUCCESS, result.message = "second stage api failed"; return finish());

      {
        const auto ret = aclrtSynchronizeStream(runtime.stream);
        CHECK_RET(ret == ACL_SUCCESS, result.statusCode = ret; result.message = "aclrtSynchronizeStream failed"; return finish());
      }

      std::vector<std::string> actualData;
      bool ok = false;
      if (!IsInplace(spec.api))
      {
        ok = ReadbackTensor(outTensor, spec.outShape, spec.outDType, &actualData);
      }
      else if (spec.api == ApiKind::kInplaceAddV3)
      {
        ok = ReadbackTensor(rhsTensor, spec.outShape, spec.outDType, &actualData);
      }
      else
      {
        ok = ReadbackTensor(lhsTensor, spec.outShape, spec.outDType, &actualData);
      }
      CHECK_RET(ok, result.statusCode = ACL_ERROR_FAILURE; result.message = "copy result from device failed"; return finish());
      result.statusCode = ACLNN_SUCCESS;
      CompareExecResult(spec, expectedData, actualData, &result);
    }
    catch (const std::exception &ex)
    {
      result.statusCode = ACL_ERROR_FAILURE;
      result.pass = false;
      result.message = ex.what();
    }
    return finish();
  }

  CaseResult RunAclnnAddApiCheck(const CaseSpec &spec, const RuntimeContext &runtime) { return RunApiCaseCore(spec, runtime); }
  CaseResult RunAclnnAddsApiCheck(const CaseSpec &spec, const RuntimeContext &runtime) { return RunApiCaseCore(spec, runtime); }
  CaseResult RunAclnnInplaceAddApiCheck(const CaseSpec &spec, const RuntimeContext &runtime) { return RunApiCaseCore(spec, runtime); }
  CaseResult RunAclnnInplaceAddsApiCheck(const CaseSpec &spec, const RuntimeContext &runtime) { return RunApiCaseCore(spec, runtime); }
  CaseResult RunAclnnAddV3ApiCheck(const CaseSpec &spec, const RuntimeContext &runtime) { return RunApiCaseCore(spec, runtime); }
  CaseResult RunAclnnInplaceAddV3ApiCheck(const CaseSpec &spec, const RuntimeContext &runtime) { return RunApiCaseCore(spec, runtime); }

  CaseResult RunAclnnAddCase(const CaseSpec &spec, const RuntimeContext &runtime) { return RunExecCaseCore(spec, runtime); }
  CaseResult RunAclnnAddsCase(const CaseSpec &spec, const RuntimeContext &runtime) { return RunExecCaseCore(spec, runtime); }
  CaseResult RunAclnnInplaceAddCase(const CaseSpec &spec, const RuntimeContext &runtime) { return RunExecCaseCore(spec, runtime); }
  CaseResult RunAclnnInplaceAddsCase(const CaseSpec &spec, const RuntimeContext &runtime) { return RunExecCaseCore(spec, runtime); }
  CaseResult RunAclnnAddV3Case(const CaseSpec &spec, const RuntimeContext &runtime) { return RunExecCaseCore(spec, runtime); }
  CaseResult RunAclnnInplaceAddV3Case(const CaseSpec &spec, const RuntimeContext &runtime) { return RunExecCaseCore(spec, runtime); }

  void PopulateApiCases(std::vector<CaseSpec> &cases)
  {
    cases.push_back({"api_add_null_lhs", "Add null lhs", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {2, 3}, {2, 3}, {}, {}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "lhs"});
    cases.push_back({"api_add_null_rhs", "Add null rhs", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {2, 3}, {2, 3}, {}, {}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "rhs"});
    cases.push_back({"api_add_null_alpha", "Add null alpha", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {2, 3}, {2, 3}, {}, {}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "alpha"});
    cases.push_back({"api_add_null_out", "Add null out", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {2, 3}, {2, 3}, {}, {}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "out"});
    cases.push_back({"api_add_out_shape_mismatch", "Add out shape mismatch", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3, 4}, {1, 3, 1}, {2, 3, 1}, {}, {}, {"1.0"}, ACLNN_ERR_PARAM_INVALID});
    cases.push_back({"api_add_mixed_invalid_out", "Add mixed dtype invalid out", ApiKind::kAdd, DType::kFloat16, DType::kFloat, DType::kFloat, DType::kFloat16, {2, 4}, {2, 4}, {2, 4}, {}, {}, {"1.0"}, ACLNN_ERR_PARAM_INVALID});
    cases.push_back({"api_add_bool_alpha_float_invalid", "Add bool promote with float alpha invalid", ApiKind::kAdd, DType::kBool, DType::kBool, DType::kFloat, DType::kBool, {2, 3}, {2, 3}, {2, 3}, {}, {}, {"1.0"}, ACLNN_ERR_PARAM_INVALID});
    cases.push_back({"api_add_dim_gt_8", "Add rank > 8 invalid", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {}, {}, {"1.0"}, ACLNN_ERR_PARAM_INVALID});
    cases.push_back({"api_add_empty_tensor_ok", "Add empty tensor success", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 0, 3}, {1, 0, 3}, {2, 0, 3}, {}, {}, {"1.0"}, ACLNN_SUCCESS});
    cases.push_back({"api_add_int32_basic", "Add int32 basic success", ApiKind::kAdd, DType::kInt32, DType::kInt32, DType::kInt32, DType::kInt32, {2, 2}, {2, 2}, {2, 2}, {}, {}, {"1"}, ACLNN_SUCCESS});
    cases.push_back({"api_add_bool_out_int32", "Add bool to int32 success", ApiKind::kAdd, DType::kBool, DType::kBool, DType::kBool, DType::kInt32, {2, 3}, {2, 3}, {2, 3}, {}, {}, {"true"}, ACLNN_SUCCESS});
    cases.push_back({"api_add_complex64_basic", "Add complex64 basic success", ApiKind::kAdd, DType::kComplex64, DType::kComplex64, DType::kFloat, DType::kComplex64, {2, 2}, {2, 2}, {2, 2}, {}, {}, {"1.0"}, ACLNN_SUCCESS});
    cases.push_back({"api_add_complex32_basic", "Add complex32 basic success", ApiKind::kAdd, DType::kComplex32, DType::kComplex32, DType::kFloat, DType::kComplex32, {2, 2}, {2, 2}, {2, 2}, {}, {}, {"1.0"}, ACLNN_SUCCESS});

    cases.push_back({"api_adds_out_shape_mismatch", "Adds out shape mismatch", ApiKind::kAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {}, {3, 2}, {}, {"1.25"}, {"1.0"}, ACLNN_ERR_PARAM_INVALID});
    cases.push_back({"api_adds_null_rhs", "Adds null scalar", ApiKind::kAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {}, {2, 3}, {}, {"1.25"}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "rhs"});
    cases.push_back({"api_adds_empty_tensor_ok", "Adds empty tensor success", ApiKind::kAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 0, 3}, {}, {2, 0, 3}, {}, {"1.25"}, {"1.0"}, ACLNN_SUCCESS});
    cases.push_back({"api_adds_null_alpha", "Adds null alpha", ApiKind::kAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {}, {2, 3}, {}, {"1.25"}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "alpha"});
    cases.push_back({"api_adds_bool_alpha_float_invalid", "Adds bool alpha float invalid", ApiKind::kAdds, DType::kBool, DType::kBool, DType::kFloat, DType::kBool, {2, 3}, {}, {2, 3}, {}, {"true"}, {"1.0"}, ACLNN_ERR_PARAM_INVALID});
    cases.push_back({"api_adds_bool_out_int32", "Adds bool to int32 success", ApiKind::kAdds, DType::kBool, DType::kBool, DType::kBool, DType::kInt32, {2, 3}, {}, {2, 3}, {}, {"true"}, {"true"}, ACLNN_SUCCESS});

    cases.push_back({"api_inplace_add_null_lhs", "InplaceAdd null lhs", ApiKind::kInplaceAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {2, 3}, {}, {}, {}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "lhs"});
    cases.push_back({"api_inplace_add_null_rhs", "InplaceAdd null rhs", ApiKind::kInplaceAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {2, 3}, {}, {}, {}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "rhs"});
    cases.push_back({"api_inplace_add_null_alpha", "InplaceAdd null alpha", ApiKind::kInplaceAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {2, 3}, {}, {}, {}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "alpha"});
    cases.push_back({"api_inplace_add_mix_invalid_other", "InplaceAdd mixed invalid target", ApiKind::kInplaceAdd, DType::kFloat16, DType::kFloat, DType::kFloat, DType::kFloat16, {2, 4}, {2, 4}, {}, {}, {}, {"1.0"}, ACLNN_ERR_PARAM_INVALID});

    cases.push_back({"api_inplace_adds_basic", "InplaceAdds basic success", ApiKind::kInplaceAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {}, {}, {}, {"1.0"}, {"1.0"}, ACLNN_SUCCESS});
    cases.push_back({"api_inplace_adds_null_alpha", "InplaceAdds null alpha", ApiKind::kInplaceAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {}, {}, {}, {"1.0"}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "alpha"});
    cases.push_back({"api_inplace_adds_bool_out_int32", "InplaceAdds bool to int32 success", ApiKind::kInplaceAdds, DType::kBool, DType::kBool, DType::kBool, DType::kInt32, {2, 3}, {}, {}, {}, {"true"}, {"true"}, ACLNN_SUCCESS});

    cases.push_back({"api_addv3_out_shape_mismatch", "AddV3 out shape mismatch", ApiKind::kAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {2, 3}, {2, 1}, {"1.5"}, {}, {"1.0"}, ACLNN_ERR_PARAM_INVALID});
    cases.push_back({"api_addv3_invalid_other_uint8", "AddV3 unsupported other uint8", ApiKind::kAddV3, DType::kFloat, DType::kUInt8, DType::kFloat, DType::kUInt8, {}, {2, 3}, {2, 3}, {"1.5"}, {}, {"1.0"}, ACLNN_ERR_PARAM_INVALID});
    cases.push_back({"api_addv3_null_self", "AddV3 null self scalar", ApiKind::kAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {2, 3}, {2, 3}, {"1.5"}, {}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "lhs"});
    cases.push_back({"api_addv3_empty_tensor_ok", "AddV3 empty tensor success", ApiKind::kAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {2, 0, 3}, {2, 0, 3}, {"1.5"}, {}, {"1.0"}, ACLNN_SUCCESS});
    cases.push_back({"api_addv3_null_alpha", "AddV3 null alpha", ApiKind::kAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {2, 3}, {2, 3}, {"1.5"}, {}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "alpha"});
    cases.push_back({"api_addv3_bool_alpha_float_invalid", "AddV3 bool alpha float invalid", ApiKind::kAddV3, DType::kBool, DType::kBool, DType::kFloat, DType::kBool, {}, {2, 3}, {2, 3}, {"true"}, {}, {"1.0"}, ACLNN_ERR_PARAM_INVALID});
    cases.push_back({"api_addv3_bool_out_int32", "AddV3 bool to int32 success", ApiKind::kAddV3, DType::kBool, DType::kBool, DType::kBool, DType::kInt32, {}, {2, 3}, {2, 3}, {"true"}, {}, {"true"}, ACLNN_SUCCESS});

    cases.push_back({"api_add_double_aicpu_basic", "Add double API planning; non-AiCore dtype route", ApiKind::kAdd, DType::kDouble, DType::kDouble, DType::kDouble, DType::kDouble, {2, 2}, {2, 2}, {2, 2}, {"1.0", "-2.0", "3.0", "-4.0"}, {"0.5", "1.5", "-3.0", "4.0"}, {"1.0"}, ACLNN_SUCCESS});
    cases.push_back({"api_add_int16_aicpu_basic", "Add int16 API planning; non-AiCore dtype route", ApiKind::kAdd, DType::kInt16, DType::kInt16, DType::kInt16, DType::kInt16, {2, 4}, {2, 4}, {2, 4}, {"1", "-2", "3", "-4", "10", "-20", "30", "-40"}, {"8", "7", "6", "5", "1", "2", "3", "4"}, {"1"}, ACLNN_SUCCESS});
    cases.push_back({"api_add_alpha_complex_to_float_invalid", "Add float alpha complex invalid cast", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kComplex64, DType::kFloat, {2, 2}, {2, 2}, {2, 2}, {"1.0", "-2.0", "3.0", "-4.0"}, {"0.5", "1.5", "-3.0", "4.0"}, {"1.0,0.0"}, ACLNN_ERR_PARAM_INVALID});
    cases.push_back({"api_add_complex64_alpha_complex", "Add complex64 with complex alpha planning", ApiKind::kAdd, DType::kComplex64, DType::kComplex64, DType::kComplex64, DType::kComplex64, {2, 2}, {2, 2}, {2, 2}, {"1.0,0.5", "-2.0,1.0", "0.0,-1.0", "3.0,4.0"}, {"0.5,-1.0", "1.0,2.0", "-1.0,0.5", "2.0,-3.0"}, {"1.0,0.0"}, ACLNN_SUCCESS});
    cases.push_back({"api_adds_int16_scalar_aicpu", "Adds int16 scalar API planning; non-AiCore dtype route", ApiKind::kAdds, DType::kInt16, DType::kInt16, DType::kInt16, DType::kInt16, {2, 4}, {}, {2, 4}, {"1", "-2", "3", "-4", "10", "-20", "30", "-40"}, {"3"}, {"1"}, ACLNN_SUCCESS});
    cases.push_back({"api_adds_double_scalar_aicpu", "Adds double scalar API planning; non-AiCore dtype route", ApiKind::kAdds, DType::kDouble, DType::kDouble, DType::kDouble, DType::kDouble, {2, 2}, {}, {2, 2}, {"1.0", "-2.0", "3.0", "-4.0"}, {"0.25"}, {"2.0"}, ACLNN_SUCCESS});
    cases.push_back({"api_adds_alpha_complex_to_float_invalid", "Adds float alpha complex invalid cast", ApiKind::kAdds, DType::kFloat, DType::kFloat, DType::kComplex64, DType::kFloat, {2, 2}, {}, {2, 2}, {"1.0", "-2.0", "3.0", "-4.0"}, {"0.5"}, {"1.0,0.0"}, ACLNN_ERR_PARAM_INVALID});
    cases.push_back({"api_adds_null_lhs", "Adds null self tensor", ApiKind::kAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {}, {2, 3}, {}, {"1.25"}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "lhs"});
    cases.push_back({"api_inplace_adds_null_lhs", "InplaceAdds null self tensor", ApiKind::kInplaceAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {}, {}, {}, {"1.0"}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "lhs"});
    cases.push_back({"api_inplace_adds_null_rhs", "InplaceAdds null scalar", ApiKind::kInplaceAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {}, {}, {}, {"1.0"}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "rhs"});
    cases.push_back({"api_inplace_add_int16_aicpu_basic", "InplaceAdd int16 API planning; non-AiCore route", ApiKind::kInplaceAdd, DType::kInt16, DType::kInt16, DType::kInt16, DType::kInt16, {2, 4}, {2, 4}, {}, {"1", "-2", "3", "-4", "10", "-20", "30", "-40"}, {"8", "7", "6", "5", "1", "2", "3", "4"}, {"1"}, ACLNN_SUCCESS});
    cases.push_back({"api_inplace_add_double_aicpu_basic", "InplaceAdd double API planning; non-AiCore route", ApiKind::kInplaceAdd, DType::kDouble, DType::kDouble, DType::kDouble, DType::kDouble, {2, 2}, {2, 2}, {}, {"1.0", "-2.0", "3.0", "-4.0"}, {"0.5", "1.5", "-3.0", "4.0"}, {"1.0"}, ACLNN_SUCCESS});
    cases.push_back({"api_inplace_add_mix_fp32_fp16_other_reject", "InplaceAdd mixed fp32/fp16 target dtype reject", ApiKind::kInplaceAdd, DType::kFloat, DType::kFloat16, DType::kFloat, DType::kFloat, {2, 4}, {2, 4}, {}, {"1.0", "-2.0", "3.0", "-4.0", "0.5", "8.0", "-16.0", "32.0"}, {"0.5", "0.25", "-1.5", "2.0", "-0.5", "1.0", "0.5", "-2.0"}, {"1.0"}, ACLNN_ERR_PARAM_INVALID});
    cases.push_back({"api_addv3_null_other", "AddV3 null other tensor", ApiKind::kAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {2, 3}, {2, 3}, {"1.5"}, {}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "rhs"});
    cases.push_back({"api_addv3_null_out", "AddV3 null out tensor", ApiKind::kAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {2, 3}, {2, 3}, {"1.5"}, {}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "out"});
    cases.push_back({"api_addv3_alpha_complex_to_float_invalid", "AddV3 float alpha complex invalid cast", ApiKind::kAddV3, DType::kFloat, DType::kFloat, DType::kComplex64, DType::kFloat, {}, {2, 3}, {2, 3}, {"1.5"}, {}, {"1.0,0.0"}, ACLNN_ERR_PARAM_INVALID});
    cases.push_back({"api_addv3_float16_alpha_fraction", "AddV3 float16 fractional alpha planning", ApiKind::kAddV3, DType::kFloat, DType::kFloat16, DType::kFloat, DType::kFloat, {}, {2, 4}, {2, 4}, {"0.25"}, {"1.0", "-2.0", "0.5", "4.0", "8.0", "-16.0", "0.125", "-0.25"}, {"0.5"}, ACLNN_SUCCESS});
    cases.push_back({"api_inplace_addv3_null_self", "InplaceAddV3 null self scalar", ApiKind::kInplaceAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {2, 3}, {}, {"1.5"}, {}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "lhs"});
    cases.push_back({"api_inplace_addv3_null_other", "InplaceAddV3 null other tensor", ApiKind::kInplaceAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {2, 3}, {}, {"1.5"}, {}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "rhs"});

    cases.push_back({"api_inplace_addv3_basic", "InplaceAddV3 basic success", ApiKind::kInplaceAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {2, 3}, {}, {"1.5"}, {}, {"1.0"}, ACLNN_SUCCESS});
    cases.push_back({"api_inplace_addv3_null_alpha", "InplaceAddV3 null alpha", ApiKind::kInplaceAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {2, 3}, {}, {"1.5"}, {}, {"1.0"}, ACLNN_ERR_PARAM_NULLPTR, "alpha"});
  }

  void PopulateExecCases(std::vector<CaseSpec> &cases)
  {
    cases.push_back({"exec_add_fp32_same_shape", "Add float32 same shape", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {2, 3}, {2, 3}, {"1.0", "-2.0", "3.5", "0.0", "-0.0", "10.0"}, {"4.0", "5.0", "-1.5", "2.0", "0.0", "-10.0"}, {"1.0"}});
    cases.push_back({"exec_add_float16_same_shape", "Add float16 same shape", ApiKind::kAdd, DType::kFloat16, DType::kFloat16, DType::kFloat, DType::kFloat16, {2, 4}, {2, 4}, {2, 4}, {"1.0", "0.5", "-2.0", "0.25", "8.0", "-4.0", "0.125", "1024.0"}, {"0.5", "1.0", "0.5", "0.25", "-2.0", "1.0", "0.125", "0.5"}, {"1.0"}});
    cases.push_back({"exec_add_mix_fp16_fp32", "Add mixed float16/float32", ApiKind::kAdd, DType::kFloat16, DType::kFloat, DType::kFloat, DType::kFloat, {2, 4}, {2, 4}, {2, 4}, {"0.33325", "1.5", "-2.25", "16.0", "0.125", "3.0", "-8.0", "0.5"}, {"1.0", "-0.5", "0.25", "2.0", "0.875", "-1.0", "0.5", "-0.25"}, {"1.0"}});
    cases.push_back({"exec_add_mix_fp32_fp16_reverse", "Add mixed float32/float16 reverse", ApiKind::kAdd, DType::kFloat, DType::kFloat16, DType::kFloat, DType::kFloat, {2, 4}, {2, 4}, {2, 4}, {"1.0", "-0.5", "0.25", "2.0", "0.875", "-1.0", "0.5", "-0.25"}, {"0.33325", "1.5", "-2.25", "16.0", "0.125", "3.0", "-8.0", "0.5"}, {"1.0"}});
    cases.push_back({"exec_add_bool_same_shape", "Add bool same shape", ApiKind::kAdd, DType::kBool, DType::kBool, DType::kBool, DType::kBool, {1, 6}, {1, 6}, {1, 6}, {"false", "true", "false", "true", "true", "false"}, {"false", "false", "true", "true", "false", "true"}, {"true"}});
    cases.push_back({"exec_add_empty_tensor", "Add empty tensor", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 0, 3}, {1, 0, 3}, {2, 0, 3}, {}, {}, {"1.0"}});
    cases.push_back({"exec_add_int32_same_shape", "Add int32 same shape", ApiKind::kAdd, DType::kInt32, DType::kInt32, DType::kInt32, DType::kInt32, {2, 4}, {2, 4}, {2, 4}, {"1", "-2", "3", "-4", "100", "-50", "0", "7"}, {"8", "7", "6", "5", "-1", "2", "3", "-7"}, {"1"}});
    cases.push_back({"exec_add_int8_same_shape", "Add int8 same shape", ApiKind::kAdd, DType::kInt8, DType::kInt8, DType::kInt8, DType::kInt8, {2, 4}, {2, 4}, {2, 4}, {"1", "-2", "3", "-4", "10", "-20", "30", "-40"}, {"8", "7", "6", "5", "-1", "2", "3", "-7"}, {"1"}});
    cases.push_back({"exec_add_uint8_same_shape", "Add uint8 same shape", ApiKind::kAdd, DType::kUInt8, DType::kUInt8, DType::kUInt8, DType::kUInt8, {2, 4}, {2, 4}, {2, 4}, {"1", "2", "3", "4", "10", "20", "30", "40"}, {"8", "7", "6", "5", "1", "2", "3", "7"}, {"1"}});
    cases.push_back({"exec_add_alpha0_fp32", "Add float32 alpha zero branch", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {2, 3}, {2, 3}, {"1.0", "-2.0", "3.5", "0.0", "4.0", "-8.0"}, {"0.5", "0.25", "-1.5", "2.0", "-0.5", "1.0"}, {"0.0"}, kAclnnInnerNullPtr});
    cases.push_back({"exec_add_alpha_neg_int32", "Add int32 alpha negative branch", ApiKind::kAdd, DType::kInt32, DType::kInt32, DType::kInt32, DType::kInt32, {2, 2}, {2, 2}, {2, 2}, {"10", "20", "-30", "40"}, {"1", "2", "3", "4"}, {"-2"}, kAclnnInnerNullPtr});
    cases.push_back({"exec_add_bool_alpha_zero", "Add bool alpha zero branch", ApiKind::kAdd, DType::kBool, DType::kBool, DType::kInt64, DType::kBool, {1, 6}, {1, 6}, {1, 6}, {"false", "true", "false", "true", "true", "false"}, {"false", "false", "true", "true", "false", "true"}, {"0"}, kAclnnInnerNullPtr});
    cases.push_back({"exec_add_int8_alpha_two", "Add int8 alpha two branch", ApiKind::kAdd, DType::kInt8, DType::kInt8, DType::kInt64, DType::kInt8, {2, 4}, {2, 4}, {2, 4}, {"1", "-2", "3", "-4", "10", "-20", "30", "-40"}, {"8", "7", "6", "5", "-1", "2", "3", "-7"}, {"2"}, kAclnnInnerNullPtr});
    cases.push_back({"exec_add_uint8_alpha_two", "Add uint8 alpha two branch", ApiKind::kAdd, DType::kUInt8, DType::kUInt8, DType::kInt64, DType::kUInt8, {2, 4}, {2, 4}, {2, 4}, {"1", "2", "3", "4", "10", "20", "30", "40"}, {"8", "7", "6", "5", "1", "2", "3", "7"}, {"2"}, kAclnnInnerNullPtr});
    cases.push_back({"exec_add_large_plus_tiny", "Add float32 significand boundary", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {1, 2}, {1, 2}, {1, 2}, {"16777216.0", "16777216.0"}, {"1.0", "2.0"}, {"1.0"}});
    cases.push_back({"exec_add_fp16_rounding_boundary", "Add float16 rounding boundary", ApiKind::kAdd, DType::kFloat16, DType::kFloat16, DType::kFloat, DType::kFloat16, {1, 4}, {1, 4}, {1, 4}, {"1.0", "1.0", "1024.0", "0.00006103515625"}, {"0.00048828125", "0.000244140625", "0.5", "0.00006103515625"}, {"1.0"}});
    cases.push_back({"exec_add_fp32_subnormal_boundary", "Add float32 subnormal boundary", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {1, 4}, {1, 4}, {1, 4}, {"1.17549435e-38", "1.40129846e-45", "1.0e-37", "1.0e-45"}, {"1.40129846e-45", "1.40129846e-45", "-1.0e-37", "1.0e-45"}, {"1.0"}});
    cases.push_back({"exec_add_fp32_cancel_near_zero", "Add float32 cancellation near zero", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {1, 6}, {1, 6}, {1, 6}, {"1.0e-37", "-1.0e-37", "1.40129846e-45", "-1.40129846e-45", "5.0e-45", "-5.0e-45"}, {"-1.0e-37", "1.0e-37", "1.40129846e-45", "1.40129846e-45", "-4.0e-45", "4.0e-45"}, {"1.0"}});
    cases.push_back({"exec_add_fp32_tiny_signed_pairs", "Add float32 tiny signed pairs", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {1, 6}, {1, 6}, {1, 6}, {"1.40129846e-45", "-1.40129846e-45", "2.80259693e-45", "-2.80259693e-45", "1.17549435e-38", "-1.17549435e-38"}, {"1.40129846e-45", "2.80259693e-45", "-1.40129846e-45", "1.40129846e-45", "1.40129846e-45", "-1.40129846e-45"}, {"1.0"}});
    cases.push_back({"exec_add_fp16_cancel_near_zero", "Add float16 cancellation near zero", ApiKind::kAdd, DType::kFloat16, DType::kFloat16, DType::kFloat, DType::kFloat16, {1, 6}, {1, 6}, {1, 6}, {"0.00006103515625", "-0.00006103515625", "0.0001220703125", "-0.0001220703125", "0.00048828125", "-0.00048828125"}, {"0.00006103515625", "0.0001220703125", "-0.00006103515625", "0.00006103515625", "-0.000244140625", "0.000244140625"}, {"1.0"}});
    cases.push_back({"exec_add_bf16_cancel_near_zero", "Add bfloat16 cancellation near zero", ApiKind::kAdd, DType::kBFloat16, DType::kBFloat16, DType::kFloat, DType::kBFloat16, {1, 6}, {1, 6}, {1, 6}, {"0.0078125", "-0.0078125", "0.015625", "-0.015625", "0.03125", "-0.03125"}, {"0.0078125", "0.015625", "-0.0078125", "0.0078125", "-0.015625", "0.015625"}, {"1.0"}, ACLNN_SUCCESS, "", true, "skip bf16 near-zero case on unsupported platforms"});
    cases.push_back({"exec_add_mix_fp32_bf16_reverse", "Add mixed float32/bfloat16 reverse", ApiKind::kAdd, DType::kFloat, DType::kBFloat16, DType::kFloat, DType::kFloat, {2, 4}, {2, 4}, {2, 4}, {"0.25", "-0.5", "1.0", "2.0", "-0.25", "1.0", "0.125", "-0.5"}, {"1.0", "0.5", "-2.0", "0.25", "8.0", "-4.0", "0.125", "256.0"}, {"1.0"}, ACLNN_SUCCESS, "", true, "skip reverse bf16 mixed path on unsupported platforms"});
    cases.push_back({"exec_add_int64_same_shape", "Add int64 same shape", ApiKind::kAdd, DType::kInt64, DType::kInt64, DType::kInt64, DType::kInt64, {2, 2}, {2, 2}, {2, 2}, {"1", "-2", "100", "-50"}, {"8", "7", "-3", "5"}, {"1"}});
    cases.push_back({"exec_add_fp64_subnormal_explore", "Add float64 subnormal explore", ApiKind::kAdd, DType::kDouble, DType::kDouble, DType::kDouble, DType::kDouble, {1, 4}, {1, 4}, {1, 4}, {"4.9406564584124654e-324", "-4.9406564584124654e-324", "9.8813129168249309e-324", "-9.8813129168249309e-324"}, {"4.9406564584124654e-324", "1.4821969375237396e-323", "-4.9406564584124654e-324", "1.4821969375237396e-323"}, {"1.0"}, ACLNN_SUCCESS, "", true, "ACL_DOUBLE may route to AICPU on simulator; keep as exploratory precision case"});
    cases.push_back({"exec_add_complex64_same_shape", "Add complex64 same shape", ApiKind::kAdd, DType::kComplex64, DType::kComplex64, DType::kFloat, DType::kComplex64, {2, 2}, {2, 2}, {2, 2}, {"1.0,2.0", "-3.0,4.0", "0.5,-0.5", "-1.5,-2.5"}, {"0.5,-1.0", "2.0,3.0", "-0.5,0.25", "1.0,-1.0"}, {"1.0"}});
    cases.push_back({"exec_add_complex32_same_shape", "Add complex32 same shape", ApiKind::kAdd, DType::kComplex32, DType::kComplex32, DType::kFloat, DType::kComplex32, {2, 2}, {2, 2}, {2, 2}, {"1.0,2.0", "-3.0,4.0", "0.5,-0.5", "-1.5,-2.5"}, {"0.5,-1.0", "2.0,3.0", "-0.5,0.25", "1.0,-1.0"}, {"1.0"}});
    cases.push_back({"exec_add_complex64_tiny_pairs", "Add complex64 tiny signed pairs", ApiKind::kAdd, DType::kComplex64, DType::kComplex64, DType::kFloat, DType::kComplex64, {1, 4}, {1, 4}, {1, 4}, {"1.40129846e-45,-1.40129846e-45", "-1.40129846e-45,1.40129846e-45", "1.0e-37,-1.0e-37", "-1.0e-37,1.0e-37"}, {"1.40129846e-45,1.40129846e-45", "1.40129846e-45,1.40129846e-45", "-1.0e-37,1.0e-37", "1.0e-37,-1.0e-37"}, {"1.0"}});
    cases.push_back({"exec_add_bool_to_int32", "Add bool to int32", ApiKind::kAdd, DType::kBool, DType::kBool, DType::kBool, DType::kInt32, {1, 6}, {1, 6}, {1, 6}, {"false", "true", "false", "true", "true", "false"}, {"false", "false", "true", "true", "false", "true"}, {"true"}});

    cases.push_back({"exec_adds_fp32_scalar", "Adds float32 scalar", ApiKind::kAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {}, {2, 3}, {"1.0", "-2.0", "3.5", "0.0", "4.0", "-8.0"}, {"1.25"}, {"1.0"}});
    cases.push_back({"exec_adds_int64_scalar", "Adds int64 scalar", ApiKind::kAdds, DType::kInt64, DType::kInt64, DType::kInt64, DType::kInt64, {2, 2}, {}, {2, 2}, {"1", "-2", "100", "-50"}, {"5"}, {"1"}});
    cases.push_back({"exec_adds_bool_special_cast", "Adds bool special cast", ApiKind::kAdds, DType::kBool, DType::kBool, DType::kBool, DType::kInt32, {1, 6}, {}, {1, 6}, {"false", "true", "false", "true", "true", "false"}, {"true"}, {"true"}, kAclnnInnerNullPtr});
    cases.push_back({"exec_adds_fp32_alpha_neg", "Adds float32 alpha negative branch", ApiKind::kAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {}, {2, 3}, {"1.0", "-2.0", "3.5", "0.0", "4.0", "-8.0"}, {"0.2"}, {"-3.0"}, kAclnnInnerNullPtr});
    cases.push_back({"exec_adds_int64_alpha_two", "Adds int64 alpha two branch", ApiKind::kAdds, DType::kInt64, DType::kInt64, DType::kInt64, DType::kInt64, {2, 2}, {}, {2, 2}, {"1", "-2", "100", "-50"}, {"5"}, {"2"}, kAclnnInnerNullPtr});
    cases.push_back({"exec_adds_int8_alpha_two", "Adds int8 alpha two branch", ApiKind::kAdds, DType::kInt8, DType::kInt8, DType::kInt64, DType::kInt8, {2, 4}, {}, {2, 4}, {"1", "-2", "3", "-4", "10", "-20", "30", "-40"}, {"3"}, {"2"}, kAclnnInnerNullPtr});
    cases.push_back({"exec_adds_int8_scalar", "Adds int8 scalar alpha one", ApiKind::kAdds, DType::kInt8, DType::kInt8, DType::kInt8, DType::kInt8, {2, 4}, {}, {2, 4}, {"1", "-2", "3", "-4", "10", "-20", "30", "-40"}, {"3"}, {"1"}});
    cases.push_back({"exec_adds_fp16_scalar_keep_b16", "Adds float16 scalar keep fp16 promote", ApiKind::kAdds, DType::kFloat16, DType::kFloat16, DType::kFloat16, DType::kFloat16, {2, 4}, {}, {2, 4}, {"1.0", "0.5", "-2.0", "0.25", "8.0", "-4.0", "0.125", "1024.0"}, {"0.5"}, {"1.0"}});
    cases.push_back({"exec_adds_fp16_scalar_promote_float", "Adds float16 scalar promote float", ApiKind::kAdds, DType::kFloat16, DType::kFloat, DType::kFloat, DType::kFloat, {2, 4}, {}, {2, 4}, {"1.0", "0.5", "-2.0", "0.25", "8.0", "-4.0", "0.125", "1024.0"}, {"0.1"}, {"1.0"}});
    cases.push_back({"exec_adds_bf16_scalar_keep_b16", "Adds bfloat16 scalar keep bf16 promote", ApiKind::kAdds, DType::kBFloat16, DType::kBFloat16, DType::kBFloat16, DType::kBFloat16, {2, 4}, {}, {2, 4}, {"1.0", "0.5", "-2.0", "0.25", "8.0", "-4.0", "0.125", "256.0"}, {"0.5"}, {"1.0"}, ACLNN_SUCCESS, "", true, "skip bf16 scalar path on unsupported platforms"});
    cases.push_back({"exec_adds_empty_tensor", "Adds empty tensor", ApiKind::kAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 0, 3}, {}, {2, 0, 3}, {}, {"1.25"}, {"1.0"}});
    cases.push_back({"exec_adds_fp32_tiny_scalar", "Adds float32 tiny scalar", ApiKind::kAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {1, 6}, {}, {1, 6}, {"1.40129846e-45", "-1.40129846e-45", "1.0e-37", "-1.0e-37", "0.0", "-0.0"}, {"1.40129846e-45"}, {"1.0"}});
    cases.push_back({"exec_adds_bool_scalar_bool_out", "Adds bool scalar to bool", ApiKind::kAdds, DType::kBool, DType::kBool, DType::kBool, DType::kBool, {1, 6}, {}, {1, 6}, {"false", "true", "false", "true", "true", "false"}, {"true"}, {"true"}});
    cases.push_back({"exec_adds_uint8_scalar", "Adds uint8 scalar alpha one", ApiKind::kAdds, DType::kUInt8, DType::kUInt8, DType::kUInt8, DType::kUInt8, {2, 4}, {}, {2, 4}, {"1", "2", "3", "4", "10", "20", "30", "40"}, {"3"}, {"1"}});
    cases.push_back({"exec_adds_int32_scalar", "Adds int32 scalar alpha one", ApiKind::kAdds, DType::kInt32, DType::kInt32, DType::kInt32, DType::kInt32, {2, 4}, {}, {2, 4}, {"1", "-2", "3", "-4", "10", "-20", "30", "-40"}, {"3"}, {"1"}});
    cases.push_back({"exec_adds_complex64_scalar", "Adds complex64 scalar", ApiKind::kAdds, DType::kComplex64, DType::kComplex64, DType::kFloat, DType::kComplex64, {2, 2}, {}, {2, 2}, {"1.0,2.0", "-3.0,4.0", "0.5,-0.5", "-1.5,-2.5"}, {"0.5,-1.0"}, {"1.0"}});
    cases.push_back({"exec_adds_complex32_scalar", "Adds complex32 scalar", ApiKind::kAdds, DType::kComplex32, DType::kComplex32, DType::kFloat, DType::kComplex32, {2, 2}, {}, {2, 2}, {"1.0,2.0", "-3.0,4.0", "0.5,-0.5", "-1.5,-2.5"}, {"0.5,-1.0"}, {"1.0"}});
    cases.push_back({"exec_adds_bf16_scalar_promote_float", "Adds bfloat16 scalar promote float", ApiKind::kAdds, DType::kBFloat16, DType::kFloat, DType::kFloat, DType::kFloat, {2, 4}, {}, {2, 4}, {"1.0", "0.5", "-2.0", "0.25", "8.0", "-4.0", "0.125", "256.0"}, {"0.1"}, {"1.0"}, kAclnnInnerNullPtr, "", true, "skip bf16 promote-float scalar path on unsupported platforms"});

    cases.push_back({"exec_inplace_add_fp32", "InplaceAdd float32", ApiKind::kInplaceAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {2, 3}, {2, 3}, {"1.0", "2.0", "3.0", "4.0", "5.0", "6.0"}, {"6.0", "5.0", "4.0", "3.0", "2.0", "1.0"}, {"1.0"}});
    cases.push_back({"exec_inplace_add_mix_fp32_fp16", "InplaceAdd mixed fp32/fp16", ApiKind::kInplaceAdd, DType::kFloat, DType::kFloat16, DType::kFloat, DType::kFloat, {2, 4}, {2, 4}, {2, 4}, {"1.0", "-2.0", "3.0", "-4.0", "0.5", "8.0", "-16.0", "32.0"}, {"0.5", "0.25", "-1.5", "2.0", "-0.5", "1.0", "0.5", "-2.0"}, {"1.0"}});
    cases.push_back({"exec_inplace_add_int64", "InplaceAdd int64", ApiKind::kInplaceAdd, DType::kInt64, DType::kInt64, DType::kInt64, DType::kInt64, {2, 2}, {2, 2}, {2, 2}, {"1", "-2", "100", "-50"}, {"8", "7", "-3", "5"}, {"1"}});
    cases.push_back({"exec_inplace_add_bool", "InplaceAdd bool", ApiKind::kInplaceAdd, DType::kBool, DType::kBool, DType::kBool, DType::kBool, {1, 6}, {1, 6}, {1, 6}, {"false", "true", "false", "true", "true", "false"}, {"false", "false", "true", "true", "false", "true"}, {"true"}});
    cases.push_back({"exec_inplace_add_uint8", "InplaceAdd uint8", ApiKind::kInplaceAdd, DType::kUInt8, DType::kUInt8, DType::kUInt8, DType::kUInt8, {2, 4}, {2, 4}, {2, 4}, {"1", "2", "3", "4", "10", "20", "30", "40"}, {"8", "7", "6", "5", "1", "2", "3", "7"}, {"1"}});
    cases.push_back({"exec_inplace_add_bf16", "InplaceAdd bfloat16", ApiKind::kInplaceAdd, DType::kBFloat16, DType::kBFloat16, DType::kFloat, DType::kBFloat16, {2, 4}, {2, 4}, {2, 4}, {"1.0", "0.5", "-2.0", "0.25", "8.0", "-4.0", "0.125", "256.0"}, {"0.5", "1.0", "0.5", "0.25", "-2.0", "1.0", "0.125", "0.5"}, {"1.0"}, ACLNN_SUCCESS, "", true, "skip bf16 inplace path on unsupported platforms"});

    cases.push_back({"exec_inplace_adds_fp32", "InplaceAdds float32", ApiKind::kInplaceAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {}, {2, 3}, {"1.0", "2.0", "3.0", "4.0", "5.0", "6.0"}, {"2.5"}, {"1.0"}});
    cases.push_back({"exec_inplace_adds_int32", "InplaceAdds int32", ApiKind::kInplaceAdds, DType::kInt32, DType::kInt32, DType::kInt32, DType::kInt32, {2, 3}, {}, {2, 3}, {"1", "2", "3", "4", "5", "6"}, {"2"}, {"1"}});
    cases.push_back({"exec_inplace_adds_alpha_neg", "InplaceAdds float alpha negative branch", ApiKind::kInplaceAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 3}, {}, {2, 3}, {"1.0", "2.0", "3.0", "4.0", "5.0", "6.0"}, {"2.5"}, {"-2.0"}, kAclnnInnerNullPtr});
    cases.push_back({"exec_inplace_adds_int8", "InplaceAdds int8", ApiKind::kInplaceAdds, DType::kInt8, DType::kInt8, DType::kInt8, DType::kInt8, {2, 4}, {}, {2, 4}, {"1", "-2", "3", "-4", "10", "-20", "30", "-40"}, {"3"}, {"1"}});
    cases.push_back({"exec_inplace_adds_bool", "InplaceAdds bool", ApiKind::kInplaceAdds, DType::kBool, DType::kBool, DType::kBool, DType::kBool, {1, 6}, {}, {1, 6}, {"false", "true", "false", "true", "true", "false"}, {"true"}, {"true"}});
    cases.push_back({"exec_inplace_adds_uint8", "InplaceAdds uint8", ApiKind::kInplaceAdds, DType::kUInt8, DType::kUInt8, DType::kUInt8, DType::kUInt8, {2, 4}, {}, {2, 4}, {"1", "2", "3", "4", "10", "20", "30", "40"}, {"3"}, {"1"}});

    cases.push_back({"exec_addv3_fp32", "AddV3 float32", ApiKind::kAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {2, 3}, {2, 3}, {"1.5"}, {"1.0", "-2.0", "3.5", "0.0", "4.0", "-8.0"}, {"1.0"}});
    cases.push_back({"exec_addv3_mix_fp32_fp16", "AddV3 mixed fp32/fp16", ApiKind::kAddV3, DType::kFloat, DType::kFloat16, DType::kFloat, DType::kFloat, {}, {2, 4}, {2, 4}, {"0.25"}, {"1.0", "-2.0", "0.5", "4.0", "8.0", "-16.0", "0.125", "-0.25"}, {"1.0"}, kAclnnInnerNullPtr});
    cases.push_back({"exec_addv3_int32", "AddV3 int32", ApiKind::kAddV3, DType::kInt32, DType::kInt32, DType::kInt32, DType::kInt32, {}, {2, 2}, {2, 2}, {"3"}, {"1", "-2", "10", "-20"}, {"1"}});
    cases.push_back({"exec_addv3_int8", "AddV3 int8", ApiKind::kAddV3, DType::kInt32, DType::kInt8, DType::kInt32, DType::kInt32, {}, {2, 4}, {2, 4}, {"3"}, {"1", "-2", "3", "-4", "10", "-20", "30", "-40"}, {"1"}});
    cases.push_back({"exec_addv3_alpha_neg", "AddV3 float alpha negative branch", ApiKind::kAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {2, 3}, {2, 3}, {"1.5"}, {"1.0", "-2.0", "3.5", "0.0", "4.0", "-8.0"}, {"-2.0"}, kAclnnInnerNullPtr});
    cases.push_back({"exec_addv3_int8_alpha_two", "AddV3 int8 alpha two fallback", ApiKind::kAddV3, DType::kInt32, DType::kInt8, DType::kInt32, DType::kInt32, {}, {2, 4}, {2, 4}, {"3"}, {"1", "-2", "3", "-4", "10", "-20", "30", "-40"}, {"2"}, kAclnnInnerNullPtr});
    cases.push_back({"exec_addv3_empty_tensor", "AddV3 empty tensor", ApiKind::kAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {2, 0, 3}, {2, 0, 3}, {"1.5"}, {}, {"1.0"}});
    cases.push_back({"exec_addv3_fp32_tiny_tensor", "AddV3 float32 tiny tensor", ApiKind::kAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {1, 6}, {1, 6}, {"1.40129846e-45"}, {"1.40129846e-45", "-1.40129846e-45", "1.0e-37", "-1.0e-37", "0.0", "-0.0"}, {"1.0"}});
    cases.push_back({"exec_addv3_bool_out_bool", "AddV3 bool to bool", ApiKind::kAddV3, DType::kBool, DType::kBool, DType::kBool, DType::kBool, {}, {1, 6}, {1, 6}, {"true"}, {"false", "true", "false", "true", "true", "false"}, {"true"}});
    cases.push_back({"exec_addv3_bool_out_int32", "AddV3 bool to int32", ApiKind::kAddV3, DType::kBool, DType::kBool, DType::kBool, DType::kInt32, {}, {1, 6}, {1, 6}, {"true"}, {"false", "true", "false", "true", "true", "false"}, {"true"}});
    cases.push_back({"exec_addv3_bf16", "AddV3 bfloat16", ApiKind::kAddV3, DType::kFloat, DType::kBFloat16, DType::kFloat, DType::kFloat, {}, {2, 4}, {2, 4}, {"1.0"}, {"1.0", "0.5", "-2.0", "0.25", "8.0", "-4.0", "0.125", "256.0"}, {"1.0"}, ACLNN_SUCCESS, "", true, "skip bf16 AddV3 on unsupported platforms"});
    cases.push_back({"exec_addv3_float16", "AddV3 float16", ApiKind::kAddV3, DType::kFloat, DType::kFloat16, DType::kFloat, DType::kFloat, {}, {2, 4}, {2, 4}, {"1.0"}, {"1.0", "0.5", "-2.0", "0.25", "8.0", "-4.0", "0.125", "1024.0"}, {"1.0"}});

    cases.push_back({"exec_inplace_addv3_fp32", "InplaceAddV3 float32", ApiKind::kInplaceAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {2, 3}, {2, 3}, {"1.5"}, {"1.0", "-2.0", "3.5", "0.0", "4.0", "-8.0"}, {"1.0"}});
    cases.push_back({"exec_inplace_addv3_int32", "InplaceAddV3 int32", ApiKind::kInplaceAddV3, DType::kInt32, DType::kInt32, DType::kInt32, DType::kInt32, {}, {2, 2}, {2, 2}, {"3"}, {"1", "-2", "10", "-20"}, {"1"}});
    cases.push_back({"exec_inplace_addv3_empty_tensor", "InplaceAddV3 empty tensor", ApiKind::kInplaceAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {2, 0, 3}, {2, 0, 3}, {"1.5"}, {}, {"1.0"}});
    cases.push_back({"prec_add_fp32_large_plus_small", "Precision/Add fp32 large + small", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {1, 4}, {1, 4}, {1, 4}, {"10000000000.0", "10000000000.0", "-10000000000.0", "-10000000000.0"}, {"0.00001", "-0.00001", "0.00001", "-0.00001"}, {"1.0"}});

    cases.push_back({"prec_add_fp32_cancellation", "Precision/Add fp32 near cancellation", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {1, 4}, {1, 4}, {1, 4}, {"1.0000001", "2.0000001", "-1.0000001", "-2.0000001"}, {"-1.0", "-2.0", "1.0", "2.0"}, {"1.0"}});

    cases.push_back({"prec_add_fp32_alpha_fraction", "Precision/Add fp32 alpha fractional scaling", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {1, 4}, {1, 4}, {1, 4}, {"1.0", "-1.0", "1024.0", "-1024.0"}, {"0.1", "0.1", "0.1", "0.1"}, {"0.1"}});

    cases.push_back({"prec_add_fp16_large_plus_small", "Precision/Add fp16 large + small", ApiKind::kAdd, DType::kFloat16, DType::kFloat16, DType::kFloat, DType::kFloat16, {1, 4}, {1, 4}, {1, 4}, {"2048.0", "2048.0", "-2048.0", "-2048.0"}, {"0.5", "1.0", "0.5", "1.0"}, {"1.0"}});

    cases.push_back({"prec_add_bf16_large_plus_small", "Precision/Add bf16 large + small", ApiKind::kAdd, DType::kBFloat16, DType::kBFloat16, DType::kFloat, DType::kBFloat16, {1, 4}, {1, 4}, {1, 4}, {"256.0", "256.0", "-256.0", "-256.0"}, {"0.5", "1.0", "0.5", "1.0"}, {"1.0"}, ACLNN_SUCCESS, "", true, "skip bf16 precision case on unsupported platforms"});

    cases.push_back({"prec_add_fp32_subnormal_pairs", "Precision/Add fp32 subnormal pairs", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {1, 6}, {1, 6}, {1, 6}, {"1.40129846e-45", "-1.40129846e-45", "2.80259693e-45", "-2.80259693e-45", "1.17549435e-38", "-1.17549435e-38"}, {"1.40129846e-45", "1.40129846e-45", "-1.40129846e-45", "1.40129846e-45", "1.40129846e-45", "-1.40129846e-45"}, {"1.0"}});

    cases.push_back({"prec_add_fp32_special_values", "Precision/Add fp32 NaN Inf propagation", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {1, 6}, {1, 6}, {1, 6}, {"nan", "inf", "-inf", "1.0", "-0.0", "0.0"}, {"1.0", "-inf", "inf", "nan", "0.0", "-0.0"}, {"1.0"}});

    cases.push_back({"prec_adds_fp32_scalar_cancel", "Precision/Adds fp32 scalar cancellation", ApiKind::kAdds, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {1, 4}, {}, {1, 4}, {"1.0000001", "-1.0000001", "10000000000.0", "-10000000000.0"}, {"1.0"}, {"-1.0"}});

    cases.push_back({"prec_inplace_add_fp32_cancel", "Precision/InplaceAdd fp32 near cancellation", ApiKind::kInplaceAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {1, 4}, {1, 4}, {1, 4}, {"1.0000001", "2.0000001", "-1.0000001", "-2.0000001"}, {"-1.0", "-2.0", "1.0", "2.0"}, {"1.0"}});

    cases.push_back({"prec_addv3_fp32_scalar_tensor_cancel", "Precision/AddV3 fp32 scalar-tensor cancellation", ApiKind::kAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {1, 4}, {1, 4}, {"1.0000001"}, {"-1.0", "-1.0000001", "1.0", "1.0000001"}, {"1.0"}});

    cases.push_back({"prec_addv3_fp32_alpha_fraction", "Precision/AddV3 fp32 fractional alpha", ApiKind::kAddV3, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {}, {1, 4}, {1, 4}, {"10000000000.0"}, {"1.0", "-1.0", "0.00001", "-0.00001"}, {"0.1"}});

    cases.push_back({"exec_add_bf16_same_shape", "Add bfloat16 same shape", ApiKind::kAdd, DType::kBFloat16, DType::kBFloat16, DType::kFloat, DType::kBFloat16, {2, 4}, {2, 4}, {2, 4}, {"1.0", "0.5", "-2.0", "0.25", "8.0", "-4.0", "0.125", "256.0"}, {"0.5", "1.0", "0.5", "0.25", "-2.0", "1.0", "0.125", "0.5"}, {"1.0"}, ACLNN_SUCCESS, "", true, "skip bf16 on platforms without advertised support"});
    cases.push_back({"exec_add_mix_bf16_fp32", "Add mixed bfloat16/float32", ApiKind::kAdd, DType::kBFloat16, DType::kFloat, DType::kFloat, DType::kFloat, {2, 4}, {2, 4}, {2, 4}, {"1.0", "0.5", "-2.0", "0.25", "8.0", "-4.0", "0.125", "256.0"}, {"0.25", "-0.5", "1.0", "2.0", "-0.25", "1.0", "0.125", "-0.5"}, {"1.0"}, ACLNN_SUCCESS, "", true, "skip bf16 mixed path on platforms without advertised support"});
  }

  void PopulateBroadcastProbeCases(std::vector<CaseSpec> &cases)
  {
    cases.push_back({"probe_add_fp32_same_shape_control", "Probe/Add fp32 same-shape control", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {"0.0", "1.0", "2.0", "3.0", "-4.0", "5.0", "-6.0", "7.0"}, {"1.0", "-1.0", "0.5", "-0.5", "2.0", "-2.0", "3.0", "-3.0"}, {"1.0"}});

    cases.push_back({"probe_add_fp32_broadcast_only", "Probe/Add fp32 broadcast only", ApiKind::kAdd, DType::kFloat, DType::kFloat, DType::kFloat, DType::kFloat, {2, 2, 2}, {1, 2, 1}, {2, 2, 2}, {"0.0", "1.0", "2.0", "3.0", "-4.0", "5.0", "-6.0", "7.0"}, {"1.0", "-2.0"}, {"1.0"}});

    cases.push_back({"probe_add_fp32_bf16_same_shape", "Probe/Add float32+bfloat16 same-shape", ApiKind::kAdd, DType::kFloat, DType::kBFloat16, DType::kFloat, DType::kFloat, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {"0.0", "1.0", "2.0", "3.0", "-4.0", "5.0", "-6.0", "7.0"}, {"1.0", "-1.0", "0.5", "-0.5", "2.0", "-2.0", "3.0", "-3.0"}, {"1.0"}, ACLNN_SUCCESS, "", true, "probe: skip bf16 mixed same-shape on platforms without advertised support"});

    cases.push_back({"probe_add_fp32_bf16_broadcast", "Probe/Add float32+bfloat16 broadcast", ApiKind::kAdd, DType::kFloat, DType::kBFloat16, DType::kFloat, DType::kFloat, {2, 2, 2}, {1, 2, 1}, {2, 2, 2}, {"0.0", "1.0", "2.0", "3.0", "-4.0", "5.0", "-6.0", "7.0"}, {"1.0", "-2.0"}, {"1.0"}, ACLNN_SUCCESS, "", true, "probe: skip bf16 mixed broadcast on platforms without advertised support"});
  }

  std::vector<CaseSpec> BuildApiCheckCases()
  {
    std::vector<CaseSpec> cases;
    PopulateApiCases(cases);
    for (auto &item : cases)
    {
      NormalizeCase(&item);
    }
    return cases;
  }

  std::vector<CaseSpec> BuildExecCases(bool includeBroadcastProbes)
  {
    std::vector<CaseSpec> cases;
    PopulateExecCases(cases);
    if (includeBroadcastProbes)
    {
      PopulateBroadcastProbeCases(cases);
    }
    for (auto &item : cases)
    {
      NormalizeCase(&item);
    }
    return cases;
  }

  CaseResult RunApiCaseByKind(const CaseSpec &spec, const RuntimeContext &runtime)
  {
    switch (spec.api)
    {
    case ApiKind::kAdd:
      return RunAclnnAddApiCheck(spec, runtime);
    case ApiKind::kAdds:
      return RunAclnnAddsApiCheck(spec, runtime);
    case ApiKind::kInplaceAdd:
      return RunAclnnInplaceAddApiCheck(spec, runtime);
    case ApiKind::kInplaceAdds:
      return RunAclnnInplaceAddsApiCheck(spec, runtime);
    case ApiKind::kAddV3:
      return RunAclnnAddV3ApiCheck(spec, runtime);
    case ApiKind::kInplaceAddV3:
      return RunAclnnInplaceAddV3ApiCheck(spec, runtime);
    }
    return MakeBaseResult(spec, Phase::kApi);
  }

  CaseResult RunExecCaseByKind(const CaseSpec &spec, const RuntimeContext &runtime)
  {
    switch (spec.api)
    {
    case ApiKind::kAdd:
      return RunAclnnAddCase(spec, runtime);
    case ApiKind::kAdds:
      return RunAclnnAddsCase(spec, runtime);
    case ApiKind::kInplaceAdd:
      return RunAclnnInplaceAddCase(spec, runtime);
    case ApiKind::kInplaceAdds:
      return RunAclnnInplaceAddsCase(spec, runtime);
    case ApiKind::kAddV3:
      return RunAclnnAddV3Case(spec, runtime);
    case ApiKind::kInplaceAddV3:
      return RunAclnnInplaceAddV3Case(spec, runtime);
    }
    return MakeBaseResult(spec, Phase::kExec);
  }

  Options ParseOptions(int argc, char **argv)
  {
    Options options;
    for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      if (arg == "--phase" && i + 1 < argc)
      {
        options.phase = argv[++i];
        continue;
      }
      if ((arg == "--case" || arg == "--case_id") && i + 1 < argc)
      {
        options.caseId = argv[++i];
        continue;
      }
      if (arg == "--device" && i + 1 < argc)
      {
        options.deviceId = std::stoi(argv[++i]);
        continue;
      }
      if (arg == "--include_broadcast_probes")
      {
        options.includeBroadcastProbes = true;
        continue;
      }
      throw std::runtime_error("unknown command line argument: " + arg);
    }
    return options;
  }

  bool MatchPhase(Phase phase, const std::string &filter)
  {
    const std::string lower = ToLower(filter);
    if (lower == "all")
    {
      return true;
    }
    if (lower == "api")
    {
      return phase == Phase::kApi;
    }
    if (lower == "exec")
    {
      return phase == Phase::kExec;
    }
    throw std::runtime_error("unsupported phase filter: " + filter);
  }

  bool IsBroadcastProbeCaseId(const std::string &caseId)
  {
    return caseId.rfind("probe_", 0) == 0;
  }

  bool MatchCaseId(const CaseSpec &spec, const std::string &caseId)
  {
    return caseId.empty() || spec.caseId == caseId;
  }

  void PrintCaseResult(const CaseResult &result)
  {
    const char *tag = result.skipped ? "SKIP" : (result.pass ? "PASS" : "FAIL");
    LOG_PRINT("[%s][%s][%s] %s status=%d message=%s\n", tag, result.phase.c_str(), result.api.c_str(),
              result.caseId.c_str(), result.statusCode, result.message.c_str());
    if (!result.expected.empty() || !result.actual.empty())
    {
      LOG_PRINT("  expect_data=%s\n", JoinTokens(result.expected).c_str());
      LOG_PRINT("  actual_data=%s\n", JoinTokens(result.actual).c_str());
    }
    if (!result.pass && !result.skipped && result.firstMismatchIndex >= 0 &&
        static_cast<size_t>(result.firstMismatchIndex) < result.expected.size() &&
        static_cast<size_t>(result.firstMismatchIndex) < result.actual.size())
    {
      LOG_PRINT("  first_mismatch=%ld expect=%s actual=%s\n", result.firstMismatchIndex,
                result.expected[static_cast<size_t>(result.firstMismatchIndex)].c_str(),
                result.actual[static_cast<size_t>(result.firstMismatchIndex)].c_str());
    }
    fflush(stdout);
  }

} // namespace

int main(int argc, char **argv)
{
  try
  {
    const Options options = ParseOptions(argc, argv);
    const std::vector<CaseSpec> apiCases = BuildApiCheckCases();
    const bool includeBroadcastProbes = options.includeBroadcastProbes || IsBroadcastProbeCaseId(options.caseId);
    const std::vector<CaseSpec> execCases = BuildExecCases(includeBroadcastProbes);

    RuntimeContext runtime;
    runtime.deviceId = options.deviceId;
    auto ret = InitRuntime(&runtime);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int total = 0;
    int passed = 0;
    int failed = 0;
    int skipped = 0;

    if (MatchPhase(Phase::kApi, options.phase))
    {
      for (const auto &spec : apiCases)
      {
        if (!MatchCaseId(spec, options.caseId))
        {
          continue;
        }
        ++total;
        const CaseResult result = RunApiCaseByKind(spec, runtime);
        PrintCaseResult(result);
        if (result.skipped)
        {
          ++skipped;
        }
        else if (result.pass)
        {
          ++passed;
        }
        else
        {
          ++failed;
        }
      }
    }

    if (MatchPhase(Phase::kExec, options.phase))
    {
      for (const auto &spec : execCases)
      {
        if (!MatchCaseId(spec, options.caseId))
        {
          continue;
        }
        ++total;
        const CaseResult result = RunExecCaseByKind(spec, runtime);
        PrintCaseResult(result);
        if (result.skipped)
        {
          ++skipped;
        }
        else if (result.pass)
        {
          ++passed;
        }
        else
        {
          ++failed;
        }
      }
    }

    LOG_PRINT("summary total=%d passed=%d failed=%d skipped=%d\n", total, passed, failed, skipped);
    FinalizeRuntime(&runtime);
    return failed == 0 ? 0 : 1;
  }
  catch (const std::exception &ex)
  {
    LOG_PRINT("fatal error: %s\n", ex.what());
    return 1;
  }
}
