/**
 * Enhanced end-to-end tests for Add operator.
 * Covers aclnnAdd / aclnnAdds / aclnnInplaceAdd / aclnnInplaceAdds /
 * aclnnAddV3 / aclnnInplaceAddV3, dtype branches, alpha branches,
 * broadcasting and precision analysis cases.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

static int g_passed = 0;
static int g_failed = 0;
static int g_precisionObserved = 0;

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return 1;
    }
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

static std::string ShapeToString(const std::vector<int64_t>& shape)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            os << ",";
        }
        os << shape[i];
    }
    os << "]";
    return os.str();
}

static void RecordResult(const std::string& name, bool ok, const std::string& detail = "")
{
    std::cout << "\nTest: " << name << "\n";
    if (!detail.empty()) {
        std::cout << detail << "\n";
    }
    if (ok) {
        ++g_passed;
        std::cout << "  [PASS]\n";
    } else {
        ++g_failed;
        std::cout << "  [FAIL]\n";
    }
}

static int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// ------------------------------ fp16 / bf16 conversion helpers ------------------------------

static uint32_t FloatToBits(float x)
{
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

static float BitsToFloat(uint32_t u)
{
    float x = 0.0f;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}

static uint16_t FloatToBf16Bits(float x)
{
    uint32_t u = FloatToBits(x);
    // Round to nearest even before truncating low 16 bits.
    uint32_t lsb = (u >> 16) & 1U;
    uint32_t roundingBias = 0x7FFFU + lsb;
    return static_cast<uint16_t>((u + roundingBias) >> 16);
}

static float Bf16BitsToFloat(uint16_t h)
{
    uint32_t u = static_cast<uint32_t>(h) << 16;
    return BitsToFloat(u);
}

static uint16_t FloatToHalfBits(float value)
{
    uint32_t f = FloatToBits(value);
    uint32_t sign = (f >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((f >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = f & 0x7FFFFFU;

    if (((f >> 23) & 0xFFU) == 0xFFU) {
        if (mant == 0) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
        return static_cast<uint16_t>(sign | 0x7E00U);
    }

    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x800000U;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t halfMant = mant >> shift;
        // round to nearest
        if ((mant >> (shift - 1)) & 1U) {
            halfMant += 1U;
        }
        return static_cast<uint16_t>(sign | halfMant);
    }

    uint32_t halfExp = static_cast<uint32_t>(exp) << 10;
    uint32_t halfMant = mant >> 13;
    if (mant & 0x1000U) {
        uint32_t rounded = (sign | halfExp | halfMant) + 1U;
        return static_cast<uint16_t>(rounded);
    }
    return static_cast<uint16_t>(sign | halfExp | halfMant);
}

static float HalfBitsToFloat(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16;
    uint32_t exp = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x03FFU;

    if (exp == 0) {
        if (mant == 0) {
            return BitsToFloat(sign);
        }
        // normalize subnormal
        exp = 1;
        while ((mant & 0x0400U) == 0) {
            mant <<= 1;
            --exp;
        }
        mant &= 0x03FFU;
        uint32_t fexp = (exp + (127 - 15)) << 23;
        uint32_t fmant = mant << 13;
        return BitsToFloat(sign | fexp | fmant);
    }

    if (exp == 0x1FU) {
        uint32_t fexp = 0xFFU << 23;
        uint32_t fmant = mant << 13;
        return BitsToFloat(sign | fexp | fmant);
    }

    uint32_t fexp = (exp + (127 - 15)) << 23;
    uint32_t fmant = mant << 13;
    return BitsToFloat(sign | fexp | fmant);
}

static uint16_t F16(float x)
{
    return FloatToHalfBits(x);
}

static uint16_t BF16(float x)
{
    return FloatToBf16Bits(x);
}

// ------------------------------ ACL tensor/scalar helpers ------------------------------

template <typename T>
static int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    int64_t elemCount = GetShapeSize(shape);
    size_t size = static_cast<size_t>(elemCount) * sizeof(T);
    *deviceAddr = nullptr;
    if (size > 0) {
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }

    std::vector<int64_t> strides(shape.size(), 1);
    if (!shape.empty()) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
        }
    }

    const int64_t* shapePtr = shape.empty() ? nullptr : shape.data();
    const int64_t* stridePtr = strides.empty() ? nullptr : strides.data();
    *tensor = aclCreateTensor(
        shapePtr, shape.size(), dataType, stridePtr, 0, aclFormat::ACL_FORMAT_ND, shapePtr, shape.size(),
        *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return -1);
    return ACL_SUCCESS;
}

template <typename T>
static int ReadDeviceTensor(const void* deviceAddr, int64_t elemCount, std::vector<T>* result)
{
    result->assign(static_cast<size_t>(elemCount), static_cast<T>(0));
    size_t size = static_cast<size_t>(elemCount) * sizeof(T);
    if (size == 0) {
        return ACL_SUCCESS;
    }
    auto ret = aclrtMemcpy(result->data(), size, deviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

struct TensorHolder {
    aclTensor* tensor = nullptr;
    void* device = nullptr;

    void Destroy()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
            tensor = nullptr;
        }
        if (device != nullptr) {
            aclrtFree(device);
            device = nullptr;
        }
    }
};

struct ScalarHolder {
    aclScalar* scalar = nullptr;
    double d = 0.0;
    float f = 0.0f;
    int64_t i64 = 0;
    int32_t i32 = 0;
    int8_t i8 = 0;
    uint8_t u8 = 0;
    bool b = false;
    uint16_t h = 0;

    aclScalar* Create(double value, aclDataType dtype)
    {
        Destroy();
        switch (dtype) {
            case aclDataType::ACL_FLOAT:
                f = static_cast<float>(value);
                scalar = aclCreateScalar(&f, dtype);
                break;
            case aclDataType::ACL_FLOAT16:
                h = FloatToHalfBits(static_cast<float>(value));
                scalar = aclCreateScalar(&h, dtype);
                break;
            case aclDataType::ACL_BF16:
                h = FloatToBf16Bits(static_cast<float>(value));
                scalar = aclCreateScalar(&h, dtype);
                break;
            case aclDataType::ACL_DOUBLE:
                d = value;
                scalar = aclCreateScalar(&d, dtype);
                break;
            case aclDataType::ACL_INT64:
                i64 = static_cast<int64_t>(value);
                scalar = aclCreateScalar(&i64, dtype);
                break;
            case aclDataType::ACL_INT32:
                i32 = static_cast<int32_t>(value);
                scalar = aclCreateScalar(&i32, dtype);
                break;
            case aclDataType::ACL_INT8:
                i8 = static_cast<int8_t>(value);
                scalar = aclCreateScalar(&i8, dtype);
                break;
            case aclDataType::ACL_UINT8:
                u8 = static_cast<uint8_t>(value);
                scalar = aclCreateScalar(&u8, dtype);
                break;
            case aclDataType::ACL_BOOL:
                b = (value != 0.0);
                scalar = aclCreateScalar(&b, dtype);
                break;
            default:
                f = static_cast<float>(value);
                scalar = aclCreateScalar(&f, aclDataType::ACL_FLOAT);
                break;
        }
        return scalar;
    }

    void Destroy()
    {
        if (scalar != nullptr) {
            aclDestroyScalar(scalar);
            scalar = nullptr;
        }
    }
};

static double ScalarValueAsDouble(double value, aclDataType dtype)
{
    switch (dtype) {
        case aclDataType::ACL_FLOAT:
            return static_cast<double>(static_cast<float>(value));
        case aclDataType::ACL_FLOAT16:
            return static_cast<double>(HalfBitsToFloat(FloatToHalfBits(static_cast<float>(value))));
        case aclDataType::ACL_BF16:
            return static_cast<double>(Bf16BitsToFloat(FloatToBf16Bits(static_cast<float>(value))));
        case aclDataType::ACL_DOUBLE:
            return value;
        case aclDataType::ACL_INT64:
            return static_cast<double>(static_cast<int64_t>(value));
        case aclDataType::ACL_INT32:
            return static_cast<double>(static_cast<int32_t>(value));
        case aclDataType::ACL_INT8:
            return static_cast<double>(static_cast<int8_t>(value));
        case aclDataType::ACL_UINT8:
            return static_cast<double>(static_cast<uint8_t>(value));
        case aclDataType::ACL_BOOL:
            return value != 0.0 ? 1.0 : 0.0;
        default:
            return value;
    }
}

template <typename T>
static double ElementAsDouble(T value, aclDataType dtype)
{
    if (dtype == aclDataType::ACL_FLOAT16) {
        return static_cast<double>(HalfBitsToFloat(static_cast<uint16_t>(value)));
    }
    if (dtype == aclDataType::ACL_BF16) {
        return static_cast<double>(Bf16BitsToFloat(static_cast<uint16_t>(value)));
    }
    if (dtype == aclDataType::ACL_BOOL) {
        return static_cast<double>(static_cast<uint8_t>(value) != 0 ? 1 : 0);
    }
    return static_cast<double>(value);
}

static double CastToOutputDouble(double value, aclDataType dtype)
{
    switch (dtype) {
        case aclDataType::ACL_FLOAT:
            return static_cast<double>(static_cast<float>(value));
        case aclDataType::ACL_FLOAT16:
            return static_cast<double>(HalfBitsToFloat(FloatToHalfBits(static_cast<float>(value))));
        case aclDataType::ACL_BF16:
            return static_cast<double>(Bf16BitsToFloat(FloatToBf16Bits(static_cast<float>(value))));
        case aclDataType::ACL_DOUBLE:
            return value;
        case aclDataType::ACL_INT64:
            return static_cast<double>(static_cast<int64_t>(value));
        case aclDataType::ACL_INT32:
            return static_cast<double>(static_cast<int32_t>(value));
        case aclDataType::ACL_INT8:
            return static_cast<double>(static_cast<int8_t>(value));
        case aclDataType::ACL_UINT8:
            return static_cast<double>(static_cast<uint8_t>(value));
        case aclDataType::ACL_BOOL:
            return value != 0.0 ? 1.0 : 0.0;
        default:
            return value;
    }
}

static bool NearlyEqual(double actual, double expected, double atol, double rtol)
{
    if (std::isnan(expected)) {
        return std::isnan(actual);
    }
    if (std::isinf(expected)) {
        return std::isinf(actual) && (std::signbit(actual) == std::signbit(expected));
    }
    if (std::isnan(actual) || std::isinf(actual)) {
        return false;
    }
    double diff = std::fabs(actual - expected);
    return diff <= (atol + rtol * std::fabs(expected));
}

static int64_t BroadcastOffset(int64_t outFlatIndex, const std::vector<int64_t>& outShape, const std::vector<int64_t>& inShape)
{
    if (inShape.empty()) {
        return 0;
    }
    int64_t offset = 0;
    int64_t stride = 1;
    int64_t tmp = outFlatIndex;
    for (int64_t od = static_cast<int64_t>(outShape.size()) - 1, id = static_cast<int64_t>(inShape.size()) - 1;
         od >= 0; --od, --id) {
        int64_t coord = tmp % outShape[static_cast<size_t>(od)];
        tmp /= outShape[static_cast<size_t>(od)];
        if (id >= 0) {
            int64_t dim = inShape[static_cast<size_t>(id)];
            int64_t usedCoord = (dim == 1) ? 0 : coord;
            offset += usedCoord * stride;
            stride *= dim;
        }
    }
    return offset;
}

static std::string PreviewVector(const std::vector<double>& v, size_t limit = 8)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(8) << "[";
    size_t n = std::min(limit, v.size());
    for (size_t i = 0; i < n; ++i) {
        if (i != 0) {
            os << ", ";
        }
        if (std::isnan(v[i])) {
            os << "nan";
        } else if (std::isinf(v[i])) {
            os << (v[i] > 0 ? "inf" : "-inf");
        } else {
            os << v[i];
        }
    }
    if (v.size() > limit) {
        os << ", ...";
    }
    os << "]";
    return os.str();
}

static bool RunPreparedExecutor(uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream,
    aclnnStatus (*runFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream), const std::string& name)
{
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::ostringstream os;
            os << "  allocate workspace failed, ret=" << ret;
            RecordResult(name, false, os.str());
            return false;
        }
    }

    auto ret = runFunc(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        std::ostringstream os;
        os << "  run failed, ret=" << ret;
        RecordResult(name, false, os.str());
        return false;
    }

    ret = aclrtSynchronizeStream(stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    if (ret != ACL_SUCCESS) {
        std::ostringstream os;
        os << "  aclrtSynchronizeStream failed, ret=" << ret;
        RecordResult(name, false, os.str());
        return false;
    }
    return true;
}

template <typename OutT>
static bool ValidateOutput(const std::string& name, const std::vector<OutT>& outHost, aclDataType outType,
    const std::vector<double>& expected, const std::vector<double>& mathExpected, double atol, double rtol,
    bool precisionCase)
{
    std::vector<double> actual(outHost.size(), 0.0);
    double maxAbsErr = 0.0;
    double maxMathErr = 0.0;
    bool ok = true;
    for (size_t i = 0; i < outHost.size(); ++i) {
        actual[i] = ElementAsDouble(outHost[i], outType);
        if (!NearlyEqual(actual[i], expected[i], atol, rtol)) {
            ok = false;
        }
        if (!std::isnan(actual[i]) && !std::isnan(expected[i]) && !std::isinf(actual[i]) && !std::isinf(expected[i])) {
            maxAbsErr = std::max(maxAbsErr, std::fabs(actual[i] - expected[i]));
        }
        if (!std::isnan(actual[i]) && !std::isnan(mathExpected[i]) && !std::isinf(actual[i]) && !std::isinf(mathExpected[i])) {
            maxMathErr = std::max(maxMathErr, std::fabs(actual[i] - mathExpected[i]));
        }
    }

    std::ostringstream os;
    os << "  Expected(dtype-rounded): " << PreviewVector(expected) << "\n";
    os << "  Actual:                 " << PreviewVector(actual) << "\n";
    os << std::scientific << std::setprecision(6);
    os << "  MaxAbsErr(vs rounded):  " << maxAbsErr << "\n";
    if (precisionCase) {
        os << "  Mathematical expected:  " << PreviewVector(mathExpected) << "\n";
        os << "  MaxAbsErr(vs math):     " << maxMathErr << "\n";
        if (maxMathErr > (atol + rtol)) {
            ++g_precisionObserved;
            os << "  [PRECISION] Device result matches dtype semantics, but differs from real-number expectation.\n";
        }
    }
    RecordResult(name, ok, os.str());
    return ok;
}

// ------------------------------ normal API cases ------------------------------

template <typename SelfT, typename OtherT, typename OutT>
static bool RunAddCase(const std::string& name, aclrtStream stream,
    const std::vector<SelfT>& selfData, const std::vector<int64_t>& selfShape, aclDataType selfType,
    const std::vector<OtherT>& otherData, const std::vector<int64_t>& otherShape, aclDataType otherType,
    double alphaValue, aclDataType alphaType,
    const std::vector<int64_t>& outShape, aclDataType outType, double atol, double rtol, bool precisionCase = false)
{
    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;
    std::vector<OutT> outInit(static_cast<size_t>(GetShapeSize(outShape)), static_cast<OutT>(0));

    int ret = CreateAclTensor(selfData, selfShape, &self.device, selfType, &self.tensor);
    if (ret != ACL_SUCCESS) {
        RecordResult(name, false, "  create self tensor failed");
        return false;
    }
    ret = CreateAclTensor(otherData, otherShape, &other.device, otherType, &other.tensor);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        RecordResult(name, false, "  create other tensor failed");
        return false;
    }
    ret = CreateAclTensor(outInit, outShape, &out.device, outType, &out.tensor);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        other.Destroy();
        RecordResult(name, false, "  create out tensor failed");
        return false;
    }
    alpha.Create(alphaValue, alphaType);
    if (alpha.scalar == nullptr) {
        self.Destroy();
        other.Destroy();
        out.Destroy();
        RecordResult(name, false, "  create alpha scalar failed");
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        other.Destroy();
        out.Destroy();
        alpha.Destroy();
        std::ostringstream os;
        os << "  aclnnAddGetWorkspaceSize failed, ret=" << ret;
        RecordResult(name, false, os.str());
        return false;
    }
    if (!RunPreparedExecutor(workspaceSize, executor, stream, aclnnAdd, name)) {
        self.Destroy();
        other.Destroy();
        out.Destroy();
        alpha.Destroy();
        return false;
    }

    std::vector<OutT> result;
    ret = ReadDeviceTensor(out.device, GetShapeSize(outShape), &result);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        other.Destroy();
        out.Destroy();
        alpha.Destroy();
        RecordResult(name, false, "  read output failed");
        return false;
    }

    double alphaDouble = ScalarValueAsDouble(alphaValue, alphaType);
    std::vector<double> expected(result.size(), 0.0);
    std::vector<double> mathExpected(result.size(), 0.0);
    for (int64_t i = 0; i < GetShapeSize(outShape); ++i) {
        int64_t si = BroadcastOffset(i, outShape, selfShape);
        int64_t oi = BroadcastOffset(i, outShape, otherShape);
        double sv = ElementAsDouble(selfData[static_cast<size_t>(si)], selfType);
        double ov = ElementAsDouble(otherData[static_cast<size_t>(oi)], otherType);
        double mathVal = sv + alphaDouble * ov;
        mathExpected[static_cast<size_t>(i)] = mathVal;
        expected[static_cast<size_t>(i)] = CastToOutputDouble(mathVal, outType);
    }

    bool ok = ValidateOutput(name, result, outType, expected, mathExpected, atol, rtol, precisionCase);
    self.Destroy();
    other.Destroy();
    out.Destroy();
    alpha.Destroy();
    return ok;
}

template <typename SelfT, typename OutT>
static bool RunAddsCase(const std::string& name, aclrtStream stream,
    const std::vector<SelfT>& selfData, const std::vector<int64_t>& selfShape, aclDataType selfType,
    double otherValue, aclDataType otherType, double alphaValue, aclDataType alphaType,
    aclDataType outType, double atol, double rtol, bool precisionCase = false,
    bool boolClampExpected = false)
{
    TensorHolder self;
    TensorHolder out;
    ScalarHolder other;
    ScalarHolder alpha;
    std::vector<OutT> outInit(static_cast<size_t>(GetShapeSize(selfShape)), static_cast<OutT>(0));

    int ret = CreateAclTensor(selfData, selfShape, &self.device, selfType, &self.tensor);
    if (ret != ACL_SUCCESS) {
        RecordResult(name, false, "  create self tensor failed");
        return false;
    }
    ret = CreateAclTensor(outInit, selfShape, &out.device, outType, &out.tensor);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        RecordResult(name, false, "  create out tensor failed");
        return false;
    }
    other.Create(otherValue, otherType);
    alpha.Create(alphaValue, alphaType);
    if (other.scalar == nullptr || alpha.scalar == nullptr) {
        self.Destroy();
        out.Destroy();
        other.Destroy();
        alpha.Destroy();
        RecordResult(name, false, "  create scalar failed");
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        out.Destroy();
        other.Destroy();
        alpha.Destroy();
        std::ostringstream os;
        os << "  aclnnAddsGetWorkspaceSize failed, ret=" << ret;
        RecordResult(name, false, os.str());
        return false;
    }
    if (!RunPreparedExecutor(workspaceSize, executor, stream, aclnnAdds, name)) {
        self.Destroy();
        out.Destroy();
        other.Destroy();
        alpha.Destroy();
        return false;
    }

    std::vector<OutT> result;
    ret = ReadDeviceTensor(out.device, GetShapeSize(selfShape), &result);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        out.Destroy();
        other.Destroy();
        alpha.Destroy();
        RecordResult(name, false, "  read output failed");
        return false;
    }

    double otherDouble = ScalarValueAsDouble(otherValue, otherType);
    double alphaDouble = ScalarValueAsDouble(alphaValue, alphaType);
    std::vector<double> expected(result.size(), 0.0);
    std::vector<double> mathExpected(result.size(), 0.0);
    for (size_t i = 0; i < result.size(); ++i) {
        double sv = ElementAsDouble(selfData[i], selfType);
        double mathVal = sv + alphaDouble * otherDouble;
        if (boolClampExpected) {
            mathVal = (mathVal != 0.0) ? 1.0 : 0.0;
        }
        mathExpected[i] = mathVal;
        expected[i] = CastToOutputDouble(mathVal, outType);
    }

    bool ok = ValidateOutput(name, result, outType, expected, mathExpected, atol, rtol, precisionCase);
    self.Destroy();
    out.Destroy();
    other.Destroy();
    alpha.Destroy();
    return ok;
}

template <typename SelfT, typename OtherT>
static bool RunInplaceAddCase(const std::string& name, aclrtStream stream,
    const std::vector<SelfT>& selfData, const std::vector<int64_t>& selfShape, aclDataType selfType,
    const std::vector<OtherT>& otherData, const std::vector<int64_t>& otherShape, aclDataType otherType,
    double alphaValue, aclDataType alphaType, double atol, double rtol)
{
    TensorHolder self;
    TensorHolder other;
    ScalarHolder alpha;

    int ret = CreateAclTensor(selfData, selfShape, &self.device, selfType, &self.tensor);
    if (ret != ACL_SUCCESS) {
        RecordResult(name, false, "  create selfRef tensor failed");
        return false;
    }
    ret = CreateAclTensor(otherData, otherShape, &other.device, otherType, &other.tensor);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        RecordResult(name, false, "  create other tensor failed");
        return false;
    }
    alpha.Create(alphaValue, alphaType);
    if (alpha.scalar == nullptr) {
        self.Destroy();
        other.Destroy();
        RecordResult(name, false, "  create alpha scalar failed");
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        other.Destroy();
        alpha.Destroy();
        std::ostringstream os;
        os << "  aclnnInplaceAddGetWorkspaceSize failed, ret=" << ret;
        RecordResult(name, false, os.str());
        return false;
    }
    if (!RunPreparedExecutor(workspaceSize, executor, stream, aclnnInplaceAdd, name)) {
        self.Destroy();
        other.Destroy();
        alpha.Destroy();
        return false;
    }

    std::vector<SelfT> result;
    ret = ReadDeviceTensor(self.device, GetShapeSize(selfShape), &result);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        other.Destroy();
        alpha.Destroy();
        RecordResult(name, false, "  read selfRef failed");
        return false;
    }

    double alphaDouble = ScalarValueAsDouble(alphaValue, alphaType);
    std::vector<double> expected(result.size(), 0.0);
    std::vector<double> mathExpected(result.size(), 0.0);
    for (int64_t i = 0; i < GetShapeSize(selfShape); ++i) {
        int64_t oi = BroadcastOffset(i, selfShape, otherShape);
        double sv = ElementAsDouble(selfData[static_cast<size_t>(i)], selfType);
        double ov = ElementAsDouble(otherData[static_cast<size_t>(oi)], otherType);
        double mathVal = sv + alphaDouble * ov;
        mathExpected[static_cast<size_t>(i)] = mathVal;
        expected[static_cast<size_t>(i)] = CastToOutputDouble(mathVal, selfType);
    }

    bool ok = ValidateOutput(name, result, selfType, expected, mathExpected, atol, rtol, false);
    self.Destroy();
    other.Destroy();
    alpha.Destroy();
    return ok;
}

template <typename SelfT>
static bool RunInplaceAddsCase(const std::string& name, aclrtStream stream,
    const std::vector<SelfT>& selfData, const std::vector<int64_t>& selfShape, aclDataType selfType,
    double otherValue, aclDataType otherType, double alphaValue, aclDataType alphaType,
    double atol, double rtol)
{
    TensorHolder self;
    ScalarHolder other;
    ScalarHolder alpha;
    int ret = CreateAclTensor(selfData, selfShape, &self.device, selfType, &self.tensor);
    if (ret != ACL_SUCCESS) {
        RecordResult(name, false, "  create selfRef tensor failed");
        return false;
    }
    other.Create(otherValue, otherType);
    alpha.Create(alphaValue, alphaType);
    if (other.scalar == nullptr || alpha.scalar == nullptr) {
        self.Destroy();
        other.Destroy();
        alpha.Destroy();
        RecordResult(name, false, "  create scalar failed");
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnInplaceAddsGetWorkspaceSize(self.tensor, other.scalar, alpha.scalar, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        other.Destroy();
        alpha.Destroy();
        std::ostringstream os;
        os << "  aclnnInplaceAddsGetWorkspaceSize failed, ret=" << ret;
        RecordResult(name, false, os.str());
        return false;
    }
    if (!RunPreparedExecutor(workspaceSize, executor, stream, aclnnInplaceAdds, name)) {
        self.Destroy();
        other.Destroy();
        alpha.Destroy();
        return false;
    }

    std::vector<SelfT> result;
    ret = ReadDeviceTensor(self.device, GetShapeSize(selfShape), &result);
    if (ret != ACL_SUCCESS) {
        self.Destroy();
        other.Destroy();
        alpha.Destroy();
        RecordResult(name, false, "  read selfRef failed");
        return false;
    }

    double otherDouble = ScalarValueAsDouble(otherValue, otherType);
    double alphaDouble = ScalarValueAsDouble(alphaValue, alphaType);
    std::vector<double> expected(result.size(), 0.0);
    std::vector<double> mathExpected(result.size(), 0.0);
    for (size_t i = 0; i < result.size(); ++i) {
        double sv = ElementAsDouble(selfData[i], selfType);
        double mathVal = sv + alphaDouble * otherDouble;
        mathExpected[i] = mathVal;
        expected[i] = CastToOutputDouble(mathVal, selfType);
    }

    bool ok = ValidateOutput(name, result, selfType, expected, mathExpected, atol, rtol, false);
    self.Destroy();
    other.Destroy();
    alpha.Destroy();
    return ok;
}

// ------------------------------ V3 API cases ------------------------------

template <typename OtherT, typename OutT>
static bool RunAddV3Case(const std::string& name, aclrtStream stream,
    double selfScalarValue, aclDataType selfScalarType,
    const std::vector<OtherT>& otherData, const std::vector<int64_t>& otherShape, aclDataType otherType,
    double alphaValue, aclDataType alphaType, aclDataType outType, double atol, double rtol,
    bool precisionCase = false)
{
    TensorHolder other;
    TensorHolder out;
    ScalarHolder selfScalar;
    ScalarHolder alpha;
    std::vector<OutT> outInit(static_cast<size_t>(GetShapeSize(otherShape)), static_cast<OutT>(0));

    int ret = CreateAclTensor(otherData, otherShape, &other.device, otherType, &other.tensor);
    if (ret != ACL_SUCCESS) {
        RecordResult(name, false, "  create other tensor failed");
        return false;
    }
    ret = CreateAclTensor(outInit, otherShape, &out.device, outType, &out.tensor);
    if (ret != ACL_SUCCESS) {
        other.Destroy();
        RecordResult(name, false, "  create out tensor failed");
        return false;
    }
    selfScalar.Create(selfScalarValue, selfScalarType);
    alpha.Create(alphaValue, alphaType);
    if (selfScalar.scalar == nullptr || alpha.scalar == nullptr) {
        other.Destroy();
        out.Destroy();
        selfScalar.Destroy();
        alpha.Destroy();
        RecordResult(name, false, "  create scalar failed");
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddV3GetWorkspaceSize(selfScalar.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        other.Destroy();
        out.Destroy();
        selfScalar.Destroy();
        alpha.Destroy();
        std::ostringstream os;
        os << "  aclnnAddV3GetWorkspaceSize failed, ret=" << ret;
        RecordResult(name, false, os.str());
        return false;
    }
    if (!RunPreparedExecutor(workspaceSize, executor, stream, aclnnAddV3, name)) {
        other.Destroy();
        out.Destroy();
        selfScalar.Destroy();
        alpha.Destroy();
        return false;
    }

    std::vector<OutT> result;
    ret = ReadDeviceTensor(out.device, GetShapeSize(otherShape), &result);
    if (ret != ACL_SUCCESS) {
        other.Destroy();
        out.Destroy();
        selfScalar.Destroy();
        alpha.Destroy();
        RecordResult(name, false, "  read output failed");
        return false;
    }

    double selfDouble = ScalarValueAsDouble(selfScalarValue, selfScalarType);
    double alphaDouble = ScalarValueAsDouble(alphaValue, alphaType);
    std::vector<double> expected(result.size(), 0.0);
    std::vector<double> mathExpected(result.size(), 0.0);
    for (size_t i = 0; i < result.size(); ++i) {
        double ov = ElementAsDouble(otherData[i], otherType);
        double mathVal = selfDouble + alphaDouble * ov;
        mathExpected[i] = mathVal;
        expected[i] = CastToOutputDouble(mathVal, outType);
    }

    bool ok = ValidateOutput(name, result, outType, expected, mathExpected, atol, rtol, precisionCase);
    other.Destroy();
    out.Destroy();
    selfScalar.Destroy();
    alpha.Destroy();
    return ok;
}

template <typename OtherT>
static bool RunInplaceAddV3Case(const std::string& name, aclrtStream stream,
    double selfScalarValue, aclDataType selfScalarType,
    const std::vector<OtherT>& otherData, const std::vector<int64_t>& otherShape, aclDataType otherType,
    double alphaValue, aclDataType alphaType, double atol, double rtol)
{
    TensorHolder other;
    ScalarHolder selfScalar;
    ScalarHolder alpha;

    int ret = CreateAclTensor(otherData, otherShape, &other.device, otherType, &other.tensor);
    if (ret != ACL_SUCCESS) {
        RecordResult(name, false, "  create other/selfRef tensor failed");
        return false;
    }
    selfScalar.Create(selfScalarValue, selfScalarType);
    alpha.Create(alphaValue, alphaType);
    if (selfScalar.scalar == nullptr || alpha.scalar == nullptr) {
        other.Destroy();
        selfScalar.Destroy();
        alpha.Destroy();
        RecordResult(name, false, "  create scalar failed");
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnInplaceAddV3GetWorkspaceSize(selfScalar.scalar, other.tensor, alpha.scalar, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        other.Destroy();
        selfScalar.Destroy();
        alpha.Destroy();
        std::ostringstream os;
        os << "  aclnnInplaceAddV3GetWorkspaceSize failed, ret=" << ret;
        RecordResult(name, false, os.str());
        return false;
    }
    if (!RunPreparedExecutor(workspaceSize, executor, stream, aclnnInplaceAddV3, name)) {
        other.Destroy();
        selfScalar.Destroy();
        alpha.Destroy();
        return false;
    }

    std::vector<OtherT> result;
    ret = ReadDeviceTensor(other.device, GetShapeSize(otherShape), &result);
    if (ret != ACL_SUCCESS) {
        other.Destroy();
        selfScalar.Destroy();
        alpha.Destroy();
        RecordResult(name, false, "  read inplace output failed");
        return false;
    }

    double selfDouble = ScalarValueAsDouble(selfScalarValue, selfScalarType);
    double alphaDouble = ScalarValueAsDouble(alphaValue, alphaType);
    std::vector<double> expected(result.size(), 0.0);
    std::vector<double> mathExpected(result.size(), 0.0);
    for (size_t i = 0; i < result.size(); ++i) {
        double ov = ElementAsDouble(otherData[i], otherType);
        double mathVal = selfDouble + alphaDouble * ov;
        mathExpected[i] = mathVal;
        expected[i] = CastToOutputDouble(mathVal, otherType);
    }

    bool ok = ValidateOutput(name, result, otherType, expected, mathExpected, atol, rtol, false);
    other.Destroy();
    selfScalar.Destroy();
    alpha.Destroy();
    return ok;
}

// ------------------------------ expected-error cases ------------------------------

static bool RunExpectedErrorCases(aclrtStream /*stream*/)
{
    bool allOk = true;

    {
        const std::string name = "Error/null self for aclnnAddGetWorkspaceSize";
        TensorHolder other;
        TensorHolder out;
        ScalarHolder alpha;
        std::vector<float> otherData = {1.0f, 2.0f};
        std::vector<float> outData = {0.0f, 0.0f};
        std::vector<int64_t> shape = {2};
        CreateAclTensor(otherData, shape, &other.device, aclDataType::ACL_FLOAT, &other.tensor);
        CreateAclTensor(outData, shape, &out.device, aclDataType::ACL_FLOAT, &out.tensor);
        alpha.Create(1.0, aclDataType::ACL_FLOAT);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnAddGetWorkspaceSize(nullptr, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        bool ok = (ret != ACL_SUCCESS);
        RecordResult(name, ok, "  Expected non-success return for nullptr self.");
        allOk = allOk && ok;
        other.Destroy();
        out.Destroy();
        alpha.Destroy();
    }

    {
        const std::string name = "Error/out shape mismatch for aclnnAdd";
        TensorHolder self;
        TensorHolder other;
        TensorHolder out;
        ScalarHolder alpha;
        std::vector<float> a = {1, 2, 3, 4};
        std::vector<float> b = {1, 1, 1, 1};
        std::vector<float> c = {0, 0};
        CreateAclTensor(a, {2, 2}, &self.device, aclDataType::ACL_FLOAT, &self.tensor);
        CreateAclTensor(b, {2, 2}, &other.device, aclDataType::ACL_FLOAT, &other.tensor);
        CreateAclTensor(c, {2}, &out.device, aclDataType::ACL_FLOAT, &out.tensor);
        alpha.Create(1.0, aclDataType::ACL_FLOAT);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        bool ok = (ret != ACL_SUCCESS);
        RecordResult(name, ok, "  Expected non-success return for wrong out shape.");
        allOk = allOk && ok;
        self.Destroy();
        other.Destroy();
        out.Destroy();
        alpha.Destroy();
    }

    {
        const std::string name = "Error/max dim > 8 for aclnnAdd";
        TensorHolder self;
        TensorHolder other;
        TensorHolder out;
        ScalarHolder alpha;
        std::vector<int64_t> shape9 = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        std::vector<float> data = {1.0f};
        CreateAclTensor(data, shape9, &self.device, aclDataType::ACL_FLOAT, &self.tensor);
        CreateAclTensor(data, shape9, &other.device, aclDataType::ACL_FLOAT, &other.tensor);
        CreateAclTensor(data, shape9, &out.device, aclDataType::ACL_FLOAT, &out.tensor);
        alpha.Create(1.0, aclDataType::ACL_FLOAT);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        bool ok = (ret != ACL_SUCCESS);
        RecordResult(name, ok, "  Expected non-success return for rank greater than MAX_DIM_LEN.");
        allOk = allOk && ok;
        self.Destroy();
        other.Destroy();
        out.Destroy();
        alpha.Destroy();
    }

    {
        const std::string name = "Error/inplace broadcast shape mismatch";
        TensorHolder self;
        TensorHolder other;
        ScalarHolder alpha;
        std::vector<float> a(6, 1.0f);
        std::vector<float> b(12, 1.0f);
        CreateAclTensor(a, {2, 3}, &self.device, aclDataType::ACL_FLOAT, &self.tensor);
        CreateAclTensor(b, {2, 2, 3}, &other.device, aclDataType::ACL_FLOAT, &other.tensor);
        alpha.Create(1.0, aclDataType::ACL_FLOAT);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, &workspaceSize, &executor);
        bool ok = (ret != ACL_SUCCESS);
        RecordResult(name, ok, "  Expected non-success return because inplace result cannot expand selfRef.");
        allOk = allOk && ok;
        self.Destroy();
        other.Destroy();
        alpha.Destroy();
    }

    {
        const std::string name = "Error/V3 out shape mismatch";
        TensorHolder other;
        TensorHolder out;
        ScalarHolder selfScalar;
        ScalarHolder alpha;
        std::vector<float> b = {1, 2, 3, 4};
        std::vector<float> c = {0, 0};
        CreateAclTensor(b, {4}, &other.device, aclDataType::ACL_FLOAT, &other.tensor);
        CreateAclTensor(c, {2}, &out.device, aclDataType::ACL_FLOAT, &out.tensor);
        selfScalar.Create(2.0, aclDataType::ACL_FLOAT);
        alpha.Create(1.0, aclDataType::ACL_FLOAT);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnAddV3GetWorkspaceSize(selfScalar.scalar, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        bool ok = (ret != ACL_SUCCESS);
        RecordResult(name, ok, "  Expected non-success return for V3 wrong out shape.");
        allOk = allOk && ok;
        other.Destroy();
        out.Destroy();
        selfScalar.Destroy();
        alpha.Destroy();
    }

    {
        const std::string name = "Error/null scalar for aclnnAdds";
        TensorHolder self;
        TensorHolder out;
        ScalarHolder alpha;
        std::vector<float> a = {1, 2};
        std::vector<float> c = {0, 0};
        CreateAclTensor(a, {2}, &self.device, aclDataType::ACL_FLOAT, &self.tensor);
        CreateAclTensor(c, {2}, &out.device, aclDataType::ACL_FLOAT, &out.tensor);
        alpha.Create(1.0, aclDataType::ACL_FLOAT);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnAddsGetWorkspaceSize(self.tensor, nullptr, alpha.scalar, out.tensor, &workspaceSize, &executor);
        bool ok = (ret != ACL_SUCCESS);
        RecordResult(name, ok, "  Expected non-success return for nullptr other scalar.");
        allOk = allOk && ok;
        self.Destroy();
        out.Destroy();
        alpha.Destroy();
    }

    return allOk;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // aclnnAdd: alpha == 1 direct Add branch, alpha != 1 Axpy branch, alpha == 0, negative alpha, broadcast.
    RunAddCase<float, float, float>(
        "aclnnAdd/float32/basic alpha=1.2 Axpy path", stream,
        {0, 1, 2, 3, 4, 5, 6, 7}, {4, 2}, aclDataType::ACL_FLOAT,
        {1, 1, 1, 2, 2, 2, 3, 3}, {4, 2}, aclDataType::ACL_FLOAT,
        1.2, aclDataType::ACL_FLOAT, {4, 2}, aclDataType::ACL_FLOAT, 1e-6, 1e-6);

    RunAddCase<float, float, float>(
        "aclnnAdd/float32/direct alpha=1", stream,
        {1.0f, -2.0f, 3.5f, 4.25f}, {4}, aclDataType::ACL_FLOAT,
        {0.5f, 2.0f, -1.5f, 0.75f}, {4}, aclDataType::ACL_FLOAT,
        1.0, aclDataType::ACL_FLOAT, {4}, aclDataType::ACL_FLOAT, 1e-6, 1e-6);

    RunAddCase<float, float, float>(
        "aclnnAdd/float32/alpha=0 broadcast other ignored", stream,
        {1, 2, 3, 4, 5, 6}, {2, 3}, aclDataType::ACL_FLOAT,
        {100, 200, 300}, {3}, aclDataType::ACL_FLOAT,
        0.0, aclDataType::ACL_FLOAT, {2, 3}, aclDataType::ACL_FLOAT, 1e-6, 1e-6);

    RunAddCase<float, float, float>(
        "aclnnAdd/float32/negative alpha 3D broadcast", stream,
        {1, 2, 3, 4, 5, 6}, {2, 1, 3}, aclDataType::ACL_FLOAT,
        {10, 20, 30, 40}, {1, 4, 1}, aclDataType::ACL_FLOAT,
        -0.5, aclDataType::ACL_FLOAT, {2, 4, 3}, aclDataType::ACL_FLOAT, 1e-6, 1e-6);

    // Larger tensor to exercise larger tiling schedule while keeping output preview short.
    {
        std::vector<float> a(4096), b(4096);
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = static_cast<float>(i % 97) * 0.25f;
            b[i] = static_cast<float>(i % 31) * -0.125f;
        }
        RunAddCase<float, float, float>(
            "aclnnAdd/float32/large tensor alpha=1", stream,
            a, {4096}, aclDataType::ACL_FLOAT, b, {4096}, aclDataType::ACL_FLOAT,
            1.0, aclDataType::ACL_FLOAT, {4096}, aclDataType::ACL_FLOAT, 1e-6, 1e-6);
    }

    // Tiling dtype branch coverage: fp16/bf16/float/int64/int32/uint8/int8/bool and mixed dtype.
    RunAddCase<uint16_t, uint16_t, uint16_t>(
        "aclnnAdd/float16/direct dtype branch", stream,
        {F16(1.0f), F16(-2.0f), F16(0.25f), F16(100.0f)}, {4}, aclDataType::ACL_FLOAT16,
        {F16(0.5f), F16(3.0f), F16(-0.25f), F16(2.0f)}, {4}, aclDataType::ACL_FLOAT16,
        1.0, aclDataType::ACL_FLOAT, {4}, aclDataType::ACL_FLOAT16, 1e-3, 1e-3);

    RunAddCase<uint16_t, uint16_t, uint16_t>(
        "aclnnAdd/bfloat16/direct dtype branch", stream,
        {BF16(1.0f), BF16(-2.0f), BF16(0.25f), BF16(100.0f)}, {4}, aclDataType::ACL_BF16,
        {BF16(0.5f), BF16(3.0f), BF16(-0.25f), BF16(2.0f)}, {4}, aclDataType::ACL_BF16,
        1.0, aclDataType::ACL_FLOAT, {4}, aclDataType::ACL_BF16, 1e-2, 1e-2);

    RunAddCase<int32_t, int32_t, int32_t>(
        "aclnnAdd/int32/exact alpha=2", stream,
        {1, -2, 100, -200}, {4}, aclDataType::ACL_INT32,
        {3, 4, -5, 6}, {4}, aclDataType::ACL_INT32,
        2.0, aclDataType::ACL_INT32, {4}, aclDataType::ACL_INT32, 0.0, 0.0);

    RunAddCase<int64_t, int64_t, int64_t>(
        "aclnnAdd/int64/direct dtype branch", stream,
        {10000000000LL, -7LL, 33LL}, {3}, aclDataType::ACL_INT64,
        {2LL, 7LL, -30LL}, {3}, aclDataType::ACL_INT64,
        1.0, aclDataType::ACL_INT64, {3}, aclDataType::ACL_INT64, 0.0, 0.0);

    RunAddCase<uint8_t, uint8_t, uint8_t>(
        "aclnnAdd/uint8/direct dtype branch", stream,
        {1, 2, 10, 100}, {4}, aclDataType::ACL_UINT8,
        {2, 3, 4, 5}, {4}, aclDataType::ACL_UINT8,
        1.0, aclDataType::ACL_UINT8, {4}, aclDataType::ACL_UINT8, 0.0, 0.0);

    RunAddCase<int8_t, int8_t, int8_t>(
        "aclnnAdd/int8/direct dtype branch", stream,
        {1, -2, 10, -30}, {4}, aclDataType::ACL_INT8,
        {2, 3, -4, 5}, {4}, aclDataType::ACL_INT8,
        1.0, aclDataType::ACL_INT8, {4}, aclDataType::ACL_INT8, 0.0, 0.0);

    RunAddCase<uint8_t, uint8_t, uint8_t>(
        "aclnnAdd/bool/direct bool branch", stream,
        {0, 1, 1, 0}, {4}, aclDataType::ACL_BOOL,
        {0, 0, 1, 1}, {4}, aclDataType::ACL_BOOL,
        1.0, aclDataType::ACL_BOOL, {4}, aclDataType::ACL_BOOL, 0.0, 0.0);

    RunAddCase<uint16_t, float, float>(
        "aclnnAdd/mixed fp16+fp32/output fp32", stream,
        {F16(1.5f), F16(-2.0f), F16(0.25f), F16(7.0f)}, {4}, aclDataType::ACL_FLOAT16,
        {0.25f, 3.5f, -0.125f, 0.5f}, {4}, aclDataType::ACL_FLOAT,
        1.0, aclDataType::ACL_FLOAT, {4}, aclDataType::ACL_FLOAT, 1e-5, 1e-5, true);

    RunAddCase<float, uint16_t, float>(
        "aclnnAdd/mixed fp32+fp16/output fp32", stream,
        {1.5f, -2.0f, 0.25f, 7.0f}, {4}, aclDataType::ACL_FLOAT,
        {F16(0.25f), F16(3.5f), F16(-0.125f), F16(0.5f)}, {4}, aclDataType::ACL_FLOAT16,
        1.0, aclDataType::ACL_FLOAT, {4}, aclDataType::ACL_FLOAT, 1e-5, 1e-5, true);

    RunAddCase<uint16_t, float, float>(
        "aclnnAdd/mixed bf16+fp32/output fp32", stream,
        {BF16(1.5f), BF16(-2.0f), BF16(0.25f), BF16(7.0f)}, {4}, aclDataType::ACL_BF16,
        {0.25f, 3.5f, -0.125f, 0.5f}, {4}, aclDataType::ACL_FLOAT,
        1.0, aclDataType::ACL_FLOAT, {4}, aclDataType::ACL_FLOAT, 1e-4, 1e-4, true);

    // aclnnAdds / aclnnInplace* API coverage.
    RunAddsCase<float, float>(
        "aclnnAdds/float32/tensor plus scalar alpha=-0.5", stream,
        {1, 2, 3, 4}, {4}, aclDataType::ACL_FLOAT,
        2.5, aclDataType::ACL_FLOAT, -0.5, aclDataType::ACL_FLOAT,
        aclDataType::ACL_FLOAT, 1e-6, 1e-6);

    RunAddsCase<int32_t, int32_t>(
        "aclnnAdds/int32/exact scalar alpha=3", stream,
        {1, -2, 3, -4}, {4}, aclDataType::ACL_INT32,
        5, aclDataType::ACL_INT32, 3, aclDataType::ACL_INT32,
        aclDataType::ACL_INT32, 0.0, 0.0);

    RunAddsCase<uint8_t, int32_t>(
        "aclnnAdds/bool scalar clamp branch output int32", stream,
        {0, 1, 0, 1}, {4}, aclDataType::ACL_BOOL,
        1, aclDataType::ACL_BOOL, 1, aclDataType::ACL_BOOL,
        aclDataType::ACL_INT32, 0.0, 0.0, false, true);

    RunInplaceAddCase<float, float>(
        "aclnnInplaceAdd/float32/broadcast alpha=0.5", stream,
        {1, 2, 3, 4, 5, 6}, {2, 3}, aclDataType::ACL_FLOAT,
        {10, 20, 30}, {3}, aclDataType::ACL_FLOAT,
        0.5, aclDataType::ACL_FLOAT, 1e-6, 1e-6);

    RunInplaceAddsCase<int32_t>(
        "aclnnInplaceAdds/int32/scalar negative alpha", stream,
        {10, 20, -30, 40}, {4}, aclDataType::ACL_INT32,
        3, aclDataType::ACL_INT32, -2, aclDataType::ACL_INT32, 0.0, 0.0);

    // V3: self is scalar, other is tensor. Cover alpha=1 Add, alpha!=1 Axpy, and non-Axpy Mul+Add for int8.
    RunAddV3Case<float, float>(
        "aclnnAddV3/float32/scalar self alpha=1", stream,
        10.0, aclDataType::ACL_FLOAT,
        {1, 2, -3, 4}, {4}, aclDataType::ACL_FLOAT,
        1.0, aclDataType::ACL_FLOAT, aclDataType::ACL_FLOAT, 1e-6, 1e-6);

    RunAddV3Case<float, float>(
        "aclnnAddV3/float32/scalar self alpha=0.25 Axpy", stream,
        -1.0, aclDataType::ACL_FLOAT,
        {4, 8, 12, 16}, {4}, aclDataType::ACL_FLOAT,
        0.25, aclDataType::ACL_FLOAT, aclDataType::ACL_FLOAT, 1e-6, 1e-6);

    RunAddV3Case<int8_t, int8_t>(
        "aclnnAddV3/int8/Mul+Add fallback path", stream,
        2, aclDataType::ACL_INT8,
        {1, -2, 3, -4}, {4}, aclDataType::ACL_INT8,
        2, aclDataType::ACL_INT8, aclDataType::ACL_INT8, 0.0, 0.0);

    RunInplaceAddV3Case<float>(
        "aclnnInplaceAddV3/float32/overwrite other", stream,
        1.5, aclDataType::ACL_FLOAT,
        {1, 2, 3, 4}, {4}, aclDataType::ACL_FLOAT,
        -1.0, aclDataType::ACL_FLOAT, 1e-6, 1e-6);

    // Precision analysis cases: pass against dtype-rounded expectation, but report real-number error.
    RunAddCase<float, float, float>(
        "Precision/float32 large plus tiny", stream,
        {1.0e10f, -1.0e10f}, {2}, aclDataType::ACL_FLOAT,
        {1.0e-5f, -1.0e-5f}, {2}, aclDataType::ACL_FLOAT,
        1.0, aclDataType::ACL_FLOAT, {2}, aclDataType::ACL_FLOAT, 1e-6, 1e-6, true);

    RunAddCase<float, float, float>(
        "Precision/float32 catastrophic cancellation", stream,
        {1.0000001f, 2.0000002f}, {2}, aclDataType::ACL_FLOAT,
        {-1.0f, -2.0f}, {2}, aclDataType::ACL_FLOAT,
        1.0, aclDataType::ACL_FLOAT, {2}, aclDataType::ACL_FLOAT, 1e-7, 1e-5, true);

    RunAddCase<float, float, float>(
        "Precision/float32 alpha fractional rounding", stream,
        {0.1f, 0.2f, 0.3f}, {3}, aclDataType::ACL_FLOAT,
        {0.2f, 0.3f, 0.4f}, {3}, aclDataType::ACL_FLOAT,
        0.1, aclDataType::ACL_FLOAT, {3}, aclDataType::ACL_FLOAT, 1e-6, 1e-6, true);

    RunAddCase<float, float, float>(
        "Special/float32 NaN and Inf propagation", stream,
        {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(), 1.0f},
        {4}, aclDataType::ACL_FLOAT,
        {1.0f, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN()},
        {4}, aclDataType::ACL_FLOAT,
        1.0, aclDataType::ACL_FLOAT, {4}, aclDataType::ACL_FLOAT, 1e-6, 1e-6);

    RunExpectedErrorCases(stream);

    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "\n========== Summary ==========" << "\n";
    std::cout << "Summary: " << g_passed << " passed, " << g_failed << " failed" << "\n";
    std::cout << "Precision observations: " << g_precisionObserved << "\n";
    std::cout << "=============================\n";

    return g_failed == 0 ? 0 : 1;
}
