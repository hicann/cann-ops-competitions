/**
 * test_aclnn_pow.cpp
 *
 */

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <algorithm>
#include <cstdio>

#include "acl/acl.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"
#include "aclnnop/aclnn_exp2.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(fmt, ...)          \
    do {                             \
        std::printf(fmt, ##__VA_ARGS__); \
    } while (0)

static int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t n = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        n *= shape[i];
    }
    return n;
}

static std::vector<int64_t> MakeStrides(const std::vector<int64_t> &shape)
{
    if (shape.empty()) {
        return std::vector<int64_t>();
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

static size_t GetDataTypeSize(aclDataType dtype)
{
    switch (dtype) {
        case ACL_BOOL:
        case ACL_INT8:
        case ACL_UINT8:
            return 1;
        case ACL_FLOAT16:
        case ACL_BF16:
        case ACL_INT16:
            return 2;
        case ACL_FLOAT:
        case ACL_INT32:
            return 4;
        case ACL_INT64:
            return 8;
        default:
            return 0;
    }
}

static bool IsFloatLike(aclDataType dtype)
{
    return dtype == ACL_FLOAT || dtype == ACL_FLOAT16 || dtype == ACL_BF16;
}

// ---------------- fp16 / bf16 convert ----------------

static uint16_t FloatToHalfBits(float f)
{
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));

    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mantissa = x & 0x007fffffu;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;

    if (((x >> 23) & 0xffu) == 0xffu) {
        if (mantissa != 0) {
            return static_cast<uint16_t>(sign | 0x7e00u);
        }
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x00800000u) >> (1 - exp);
        if (mantissa & 0x00001000u) {
            mantissa += 0x00002000u;
        }
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }

    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    if (mantissa & 0x00001000u) {
        mantissa += 0x00002000u;
        if (mantissa & 0x00800000u) {
            mantissa = 0;
            ++exp;
            if (exp >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mantissa >> 13));
}

static float HalfBitsToFloat(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mantissa = h & 0x03ffu;

    uint32_t out = 0;
    if (exp == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exp;
            }
            mantissa &= 0x03ffu;
            exp = exp + 127 - 15;
            out = sign | (exp << 23) | (mantissa << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mantissa << 13);
    } else {
        exp = exp + 127 - 15;
        out = sign | (exp << 23) | (mantissa << 13);
    }

    float f = 0.0f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

static uint16_t FloatToBf16Bits(float f)
{
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t lsb = (x >> 16) & 1u;
    x += 0x7fffu + lsb;
    return static_cast<uint16_t>(x >> 16);
}

static float Bf16BitsToFloat(uint16_t b)
{
    uint32_t x = static_cast<uint32_t>(b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

// ---------------- value read / write ----------------

static double ReadValueAsDouble(const std::vector<uint8_t> &buf, aclDataType dtype, int64_t idx)
{
    const size_t elemSize = GetDataTypeSize(dtype);
    const uint8_t *p = buf.data() + static_cast<size_t>(idx) * elemSize;

    switch (dtype) {
        case ACL_BOOL: {
            uint8_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<double>(v != 0);
        }
        case ACL_INT8: {
            int8_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_UINT8: {
            uint8_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_INT16: {
            int16_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_INT32: {
            int32_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_INT64: {
            int64_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_FLOAT: {
            float v = 0.0f;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_FLOAT16: {
            uint16_t bits = 0;
            std::memcpy(&bits, p, sizeof(bits));
            return static_cast<double>(HalfBitsToFloat(bits));
        }
        case ACL_BF16: {
            uint16_t bits = 0;
            std::memcpy(&bits, p, sizeof(bits));
            return static_cast<double>(Bf16BitsToFloat(bits));
        }
        default:
            return 0.0;
    }
}

static void WriteValueFromDouble(std::vector<uint8_t> &buf, aclDataType dtype, int64_t idx, double value)
{
    const size_t elemSize = GetDataTypeSize(dtype);
    uint8_t *p = buf.data() + static_cast<size_t>(idx) * elemSize;

    switch (dtype) {
        case ACL_BOOL: {
            uint8_t v = (value != 0.0) ? 1 : 0;
            std::memcpy(p, &v, sizeof(v));
            break;
        }
        case ACL_INT8: {
            int8_t v = static_cast<int8_t>(std::llround(value));
            std::memcpy(p, &v, sizeof(v));
            break;
        }
        case ACL_UINT8: {
            uint8_t v = static_cast<uint8_t>(std::llround(value));
            std::memcpy(p, &v, sizeof(v));
            break;
        }
        case ACL_INT16: {
            int16_t v = static_cast<int16_t>(std::llround(value));
            std::memcpy(p, &v, sizeof(v));
            break;
        }
        case ACL_INT32: {
            int32_t v = static_cast<int32_t>(std::llround(value));
            std::memcpy(p, &v, sizeof(v));
            break;
        }
        case ACL_INT64: {
            int64_t v = static_cast<int64_t>(std::llround(value));
            std::memcpy(p, &v, sizeof(v));
            break;
        }
        case ACL_FLOAT: {
            float v = static_cast<float>(value);
            std::memcpy(p, &v, sizeof(v));
            break;
        }
        case ACL_FLOAT16: {
            float v = static_cast<float>(value);
            uint16_t bits = FloatToHalfBits(v);
            std::memcpy(p, &bits, sizeof(bits));
            break;
        }
        case ACL_BF16: {
            float v = static_cast<float>(value);
            uint16_t bits = FloatToBf16Bits(v);
            std::memcpy(p, &bits, sizeof(bits));
            break;
        }
        default:
            break;
    }
}

static double CastLikeDtype(aclDataType dtype, double value)
{
    std::vector<uint8_t> tmp(GetDataTypeSize(dtype), 0);
    WriteValueFromDouble(tmp, dtype, 0, value);
    return ReadValueAsDouble(tmp, dtype, 0);
}

// ---------------- specs ----------------

struct TensorSpec {
    aclDataType dtype;
    std::vector<int64_t> shape;
    std::vector<double> values;
};

struct ScalarSpec {
    aclDataType dtype;
    double value;
};

static std::vector<double> MakeRamp(int64_t n, double start, double step)
{
    std::vector<double> v(static_cast<size_t>(n), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        v[static_cast<size_t>(i)] = start + step * static_cast<double>(i);
    }
    return v;
}

static std::vector<uint8_t> PackTensor(const TensorSpec &spec)
{
    int64_t n = GetShapeSize(spec.shape);
    std::vector<uint8_t> bytes(static_cast<size_t>(n) * GetDataTypeSize(spec.dtype), 0);
    for (int64_t i = 0; i < n; ++i) {
        WriteValueFromDouble(bytes, spec.dtype, i, spec.values[static_cast<size_t>(i)]);
    }
    return bytes;
}

static std::vector<uint8_t> PackScalar(const ScalarSpec &spec)
{
    std::vector<uint8_t> bytes(GetDataTypeSize(spec.dtype), 0);
    WriteValueFromDouble(bytes, spec.dtype, 0, spec.value);
    return bytes;
}

// ---------------- acl holders ----------------

struct TensorHolder {
    aclDataType dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> hostBytes;
    void *deviceAddr;
    aclTensor *tensor;

    TensorHolder() : dtype(ACL_FLOAT), deviceAddr(nullptr), tensor(nullptr) {}

    int Create(const TensorSpec &spec)
    {
        dtype = spec.dtype;
        shape = spec.shape;
        hostBytes = PackTensor(spec);
        return CreateInternal();
    }

    int CreateZero(aclDataType dataType, const std::vector<int64_t> &tensorShape)
    {
        dtype = dataType;
        shape = tensorShape;
        int64_t n = GetShapeSize(shape);
        hostBytes.assign(static_cast<size_t>(n) * GetDataTypeSize(dtype), 0);
        return CreateInternal();
    }

    int CopyBack()
    {
        CHECK_RET(deviceAddr != nullptr, return -1);
        auto ret = aclrtMemcpy(
            hostBytes.data(), hostBytes.size(),
            deviceAddr, hostBytes.size(),
            ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        return ACL_SUCCESS;
    }

    void Destroy()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
            tensor = nullptr;
        }
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
            deviceAddr = nullptr;
        }
    }

private:
    int CreateInternal()
    {
        size_t size = hostBytes.size();
        auto ret = aclrtMalloc(&deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = aclrtMemcpy(deviceAddr, size, hostBytes.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        std::vector<int64_t> strides = MakeStrides(shape);
        tensor = aclCreateTensor(
            shape.data(),
            shape.size(),
            dtype,
            strides.data(),
            0,
            aclFormat::ACL_FORMAT_ND,
            shape.data(),
            shape.size(),
            deviceAddr);
        CHECK_RET(tensor != nullptr, return -2);
        return ACL_SUCCESS;
    }
};

struct ScalarHolder {
    aclDataType dtype;
    std::vector<uint8_t> hostBytes;
    aclScalar *scalar;

    ScalarHolder() : dtype(ACL_FLOAT), scalar(nullptr) {}

    int Create(const ScalarSpec &spec)
    {
        dtype = spec.dtype;
        hostBytes = PackScalar(spec);
        scalar = aclCreateScalar(hostBytes.data(), dtype);
        CHECK_RET(scalar != nullptr, return -1);
        return ACL_SUCCESS;
    }

    void Destroy()
    {
        if (scalar != nullptr) {
            aclDestroyScalar(scalar);
            scalar = nullptr;
        }
    }
};

// ---------------- broadcast helpers ----------------

static std::vector<int64_t> BroadcastShape(const std::vector<int64_t> &a, const std::vector<int64_t> &b)
{
    int64_t ra = static_cast<int64_t>(a.size());
    int64_t rb = static_cast<int64_t>(b.size());
    int64_t r = std::max(ra, rb);

    std::vector<int64_t> out(static_cast<size_t>(r), 1);
    for (int64_t i = 0; i < r; ++i) {
        int64_t da = (i < r - ra) ? 1 : a[static_cast<size_t>(i - (r - ra))];
        int64_t db = (i < r - rb) ? 1 : b[static_cast<size_t>(i - (r - rb))];
        out[static_cast<size_t>(i)] = std::max(da, db);
    }
    return out;
}

static std::vector<int64_t> UnravelIndex(int64_t idx, const std::vector<int64_t> &shape)
{
    std::vector<int64_t> coord(shape.size(), 0);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        coord[static_cast<size_t>(i)] = idx % shape[static_cast<size_t>(i)];
        idx /= shape[static_cast<size_t>(i)];
    }
    return coord;
}

static int64_t BroadcastOffset(const std::vector<int64_t> &outCoord, const std::vector<int64_t> &inShape)
{
    if (inShape.empty()) {
        return 0;
    }

    std::vector<int64_t> strides = MakeStrides(inShape);
    int64_t outRank = static_cast<int64_t>(outCoord.size());
    int64_t inRank = static_cast<int64_t>(inShape.size());
    int64_t offset = 0;

    for (int64_t i = 0; i < inRank; ++i) {
        int64_t outDim = outRank - inRank + i;
        int64_t c = (inShape[static_cast<size_t>(i)] == 1) ? 0 : outCoord[static_cast<size_t>(outDim)];
        offset += c * strides[static_cast<size_t>(i)];
    }
    return offset;
}

// ---------------- expected ----------------

static double PowDouble(double base, double exponent)
{
    return std::pow(base, exponent);
}

static std::vector<double> ExpectedPowTensorScalar(
    const TensorSpec &selfSpec,
    const ScalarSpec &expSpec,
    aclDataType outDtype,
    const std::vector<int64_t> &outShape)
{
    std::vector<uint8_t> selfBytes = PackTensor(selfSpec);
    double exponent = CastLikeDtype(expSpec.dtype, expSpec.value);

    int64_t outNumel = GetShapeSize(outShape);
    std::vector<double> expected(static_cast<size_t>(outNumel), 0.0);

    for (int64_t i = 0; i < outNumel; ++i) {
        std::vector<int64_t> coord = UnravelIndex(i, outShape);
        int64_t aoff = BroadcastOffset(coord, selfSpec.shape);
        double base = ReadValueAsDouble(selfBytes, selfSpec.dtype, aoff);
        expected[static_cast<size_t>(i)] = CastLikeDtype(outDtype, PowDouble(base, exponent));
    }
    return expected;
}

static std::vector<double> ExpectedPowScalarTensor(
    const ScalarSpec &selfSpec,
    const TensorSpec &expSpec,
    aclDataType outDtype,
    const std::vector<int64_t> &outShape)
{
    double base = CastLikeDtype(selfSpec.dtype, selfSpec.value);
    std::vector<uint8_t> expBytes = PackTensor(expSpec);

    int64_t outNumel = GetShapeSize(outShape);
    std::vector<double> expected(static_cast<size_t>(outNumel), 0.0);

    for (int64_t i = 0; i < outNumel; ++i) {
        std::vector<int64_t> coord = UnravelIndex(i, outShape);
        int64_t eoff = BroadcastOffset(coord, expSpec.shape);
        double exponent = ReadValueAsDouble(expBytes, expSpec.dtype, eoff);
        expected[static_cast<size_t>(i)] = CastLikeDtype(outDtype, PowDouble(base, exponent));
    }
    return expected;
}

static std::vector<double> ExpectedPowTensorTensor(
    const TensorSpec &selfSpec,
    const TensorSpec &expSpec,
    aclDataType outDtype,
    const std::vector<int64_t> &outShape)
{
    std::vector<uint8_t> selfBytes = PackTensor(selfSpec);
    std::vector<uint8_t> expBytes = PackTensor(expSpec);

    int64_t outNumel = GetShapeSize(outShape);
    std::vector<double> expected(static_cast<size_t>(outNumel), 0.0);

    for (int64_t i = 0; i < outNumel; ++i) {
        std::vector<int64_t> coord = UnravelIndex(i, outShape);
        int64_t boff = BroadcastOffset(coord, selfSpec.shape);
        int64_t eoff = BroadcastOffset(coord, expSpec.shape);
        double base = ReadValueAsDouble(selfBytes, selfSpec.dtype, boff);
        double exponent = ReadValueAsDouble(expBytes, expSpec.dtype, eoff);
        expected[static_cast<size_t>(i)] = CastLikeDtype(outDtype, PowDouble(base, exponent));
    }
    return expected;
}

static std::vector<double> ExpectedExp2(
    const TensorSpec &selfSpec,
    aclDataType outDtype,
    const std::vector<int64_t> &outShape)
{
    std::vector<uint8_t> selfBytes = PackTensor(selfSpec);
    int64_t outNumel = GetShapeSize(outShape);
    std::vector<double> expected(static_cast<size_t>(outNumel), 0.0);

    for (int64_t i = 0; i < outNumel; ++i) {
        std::vector<int64_t> coord = UnravelIndex(i, outShape);
        int64_t off = BroadcastOffset(coord, selfSpec.shape);
        double x = ReadValueAsDouble(selfBytes, selfSpec.dtype, off);
        expected[static_cast<size_t>(i)] = CastLikeDtype(outDtype, PowDouble(2.0, x));
    }
    return expected;
}

// ---------------- compare ----------------

static bool CompareTensor(
    const TensorHolder &actual,
    const std::vector<double> &expected,
    double atol,
    double rtol,
    std::string &msg)
{
    int64_t n = GetShapeSize(actual.shape);
    if (static_cast<int64_t>(expected.size()) != n) {
        msg = "expected size mismatch";
        return false;
    }

    for (int64_t i = 0; i < n; ++i) {
        double a = ReadValueAsDouble(actual.hostBytes, actual.dtype, i);
        double e = expected[static_cast<size_t>(i)];

        if (std::isnan(e)) {
            if (!std::isnan(a)) {
                std::ostringstream oss;
                oss << "idx=" << i << " expect NaN, got " << a;
                msg = oss.str();
                return false;
            }
            continue;
        }

        if (std::isinf(e)) {
            if (!(std::isinf(a) && (std::signbit(a) == std::signbit(e)))) {
                std::ostringstream oss;
                oss << "idx=" << i << " expect Inf, got " << a;
                msg = oss.str();
                return false;
            }
            continue;
        }

        if (IsFloatLike(actual.dtype)) {
            double diff = std::fabs(a - e);
            double thr = atol + rtol * std::fabs(e);
            if (diff > thr) {
                std::ostringstream oss;
                oss << "idx=" << i << " actual=" << a << " expect=" << e
                    << " diff=" << diff << " thr=" << thr;
                msg = oss.str();
                return false;
            }
        } else {
            if (a != e) {
                std::ostringstream oss;
                oss << "idx=" << i << " actual=" << a << " expect=" << e;
                msg = oss.str();
                return false;
            }
        }
    }

    msg = "ok";
    return true;
}

// ---------------- runtime ----------------

static int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);

    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);

    return ACL_SUCCESS;
}

static void Finalize(int32_t deviceId, aclrtStream stream)
{
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

// ---------------- op wrappers ----------------

static int RunPowTensorScalar(
    aclTensor *self, aclScalar *exponent, aclTensor *out, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    auto ret = aclnnPowTensorScalarGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    ret = aclrtSynchronizeStream(stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

static int RunInplacePowTensorScalar(
    aclTensor *selfRef, aclScalar *exponent, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    auto ret = aclnnInplacePowTensorScalarGetWorkspaceSize(selfRef, exponent, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnInplacePowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    ret = aclrtSynchronizeStream(stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

static int RunPowScalarTensor(
    aclScalar *self, aclTensor *exponent, aclTensor *out, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    auto ret = aclnnPowScalarTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnPowScalarTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    ret = aclrtSynchronizeStream(stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

static int RunPowTensorTensor(
    aclTensor *self, aclTensor *exponent, aclTensor *out, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    auto ret = aclnnPowTensorTensorGetWorkspaceSize(self, exponent, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnPowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    ret = aclrtSynchronizeStream(stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

static int RunInplacePowTensorTensor(
    aclTensor *selfRef, aclTensor *exponent, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    auto ret = aclnnInplacePowTensorTensorGetWorkspaceSize(selfRef, exponent, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnInplacePowTensorTensor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    ret = aclrtSynchronizeStream(stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

static int RunExp2(
    aclTensor *self, aclTensor *out, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    auto ret = aclnnExp2GetWorkspaceSize(self, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnExp2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    ret = aclrtSynchronizeStream(stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

static int RunInplaceExp2(
    aclTensor *selfRef, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    auto ret = aclnnInplaceExp2GetWorkspaceSize(selfRef, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnInplaceExp2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    ret = aclrtSynchronizeStream(stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

// ---------------- case runner ----------------

enum ApiKind {
    API_POW_TENSOR_SCALAR = 0,
    API_INPLACE_POW_TENSOR_SCALAR = 1,
    API_POW_SCALAR_TENSOR = 2,
    API_POW_TENSOR_TENSOR = 3,
    API_INPLACE_POW_TENSOR_TENSOR = 4,
    API_EXP2 = 5,
    API_INPLACE_EXP2 = 6
};

static bool RunCase(
    const std::string &name,
    ApiKind apiKind,
    aclrtStream stream,
    const TensorSpec *selfTensorSpec,
    const ScalarSpec *selfScalarSpec,
    const TensorSpec *expTensorSpec,
    const ScalarSpec *expScalarSpec,
    aclDataType outDtype,
    const std::vector<int64_t> &outShape,
    double atol,
    double rtol)
{
    TensorHolder selfTensor;
    TensorHolder expTensor;
    TensorHolder outTensor;
    ScalarHolder selfScalar;
    ScalarHolder expScalar;

    int ret = ACL_SUCCESS;

    if (selfTensorSpec != nullptr) {
        ret = selfTensor.Create(*selfTensorSpec);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s | create self tensor failed ret=%d\n", name.c_str(), ret); return false);
    }

    if (expTensorSpec != nullptr) {
        ret = expTensor.Create(*expTensorSpec);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s | create exp tensor failed ret=%d\n", name.c_str(), ret);
                  selfTensor.Destroy(); return false);
    }

    if (selfScalarSpec != nullptr) {
        ret = selfScalar.Create(*selfScalarSpec);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s | create self scalar failed ret=%d\n", name.c_str(), ret);
                  selfTensor.Destroy(); expTensor.Destroy(); return false);
    }

    if (expScalarSpec != nullptr) {
        ret = expScalar.Create(*expScalarSpec);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s | create exp scalar failed ret=%d\n", name.c_str(), ret);
                  selfTensor.Destroy(); expTensor.Destroy(); selfScalar.Destroy(); return false);
    }

    if (apiKind == API_POW_TENSOR_SCALAR ||
        apiKind == API_POW_SCALAR_TENSOR ||
        apiKind == API_POW_TENSOR_TENSOR ||
        apiKind == API_EXP2) {
        ret = outTensor.CreateZero(outDtype, outShape);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s | create out failed ret=%d\n", name.c_str(), ret);
                  selfTensor.Destroy(); expTensor.Destroy(); selfScalar.Destroy(); expScalar.Destroy(); return false);
    }

    switch (apiKind) {
        case API_POW_TENSOR_SCALAR:
            ret = RunPowTensorScalar(selfTensor.tensor, expScalar.scalar, outTensor.tensor, stream);
            break;
        case API_INPLACE_POW_TENSOR_SCALAR:
            ret = RunInplacePowTensorScalar(selfTensor.tensor, expScalar.scalar, stream);
            break;
        case API_POW_SCALAR_TENSOR:
            ret = RunPowScalarTensor(selfScalar.scalar, expTensor.tensor, outTensor.tensor, stream);
            break;
        case API_POW_TENSOR_TENSOR:
            ret = RunPowTensorTensor(selfTensor.tensor, expTensor.tensor, outTensor.tensor, stream);
            break;
        case API_INPLACE_POW_TENSOR_TENSOR:
            ret = RunInplacePowTensorTensor(selfTensor.tensor, expTensor.tensor, stream);
            break;
        case API_EXP2:
            ret = RunExp2(selfTensor.tensor, outTensor.tensor, stream);
            break;
        case API_INPLACE_EXP2:
            ret = RunInplaceExp2(selfTensor.tensor, stream);
            break;
        default:
            ret = -9999;
            break;
    }

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s | execute failed ret=%d\n", name.c_str(), ret);
        selfTensor.Destroy();
        expTensor.Destroy();
        outTensor.Destroy();
        selfScalar.Destroy();
        expScalar.Destroy();
        return false;
    }

    std::vector<double> expected;
    TensorHolder *actual = nullptr;

    switch (apiKind) {
        case API_POW_TENSOR_SCALAR:
            expected = ExpectedPowTensorScalar(*selfTensorSpec, *expScalarSpec, outDtype, outShape);
            actual = &outTensor;
            break;
        case API_INPLACE_POW_TENSOR_SCALAR:
            expected = ExpectedPowTensorScalar(*selfTensorSpec, *expScalarSpec, selfTensorSpec->dtype, selfTensorSpec->shape);
            actual = &selfTensor;
            break;
        case API_POW_SCALAR_TENSOR:
            expected = ExpectedPowScalarTensor(*selfScalarSpec, *expTensorSpec, outDtype, outShape);
            actual = &outTensor;
            break;
        case API_POW_TENSOR_TENSOR:
            expected = ExpectedPowTensorTensor(*selfTensorSpec, *expTensorSpec, outDtype, outShape);
            actual = &outTensor;
            break;
        case API_INPLACE_POW_TENSOR_TENSOR:
            expected = ExpectedPowTensorTensor(*selfTensorSpec, *expTensorSpec, selfTensorSpec->dtype, selfTensorSpec->shape);
            actual = &selfTensor;
            break;
        case API_EXP2:
            expected = ExpectedExp2(*selfTensorSpec, outDtype, outShape);
            actual = &outTensor;
            break;
        case API_INPLACE_EXP2:
            expected = ExpectedExp2(*selfTensorSpec, selfTensorSpec->dtype, selfTensorSpec->shape);
            actual = &selfTensor;
            break;
        default:
            break;
    }

    ret = actual->CopyBack();
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s | copy back failed ret=%d\n", name.c_str(), ret);
        selfTensor.Destroy();
        expTensor.Destroy();
        outTensor.Destroy();
        selfScalar.Destroy();
        expScalar.Destroy();
        return false;
    }

    std::string msg;
    bool ok = CompareTensor(*actual, expected, atol, rtol, msg);
    if (ok) {
        LOG_PRINT("[PASS] %s\n", name.c_str());
    } else {
        LOG_PRINT("[FAIL] %s | %s\n", name.c_str(), msg.c_str());
    }

    selfTensor.Destroy();
    expTensor.Destroy();
    outTensor.Destroy();
    selfScalar.Destroy();
    expScalar.Destroy();
    return ok;
}

// ---------------- negative cases ----------------

static bool NegCase_PowTensorScalar_NullSelf()
{
    TensorHolder out;
    ScalarHolder exponent;

    int ret = out.CreateZero(ACL_FLOAT, std::vector<int64_t>{2, 2});
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_PowTensorScalar_NullSelf | create out failed ret=%d\n", ret); return false);

    ScalarSpec exponentSpec;
    exponentSpec.dtype = ACL_FLOAT;
    exponentSpec.value = 2.0;
    ret = exponent.Create(exponentSpec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_PowTensorScalar_NullSelf | create exponent failed ret=%d\n", ret);
              out.Destroy(); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto apiRet = aclnnPowTensorScalarGetWorkspaceSize(nullptr, exponent.scalar, out.tensor, &workspaceSize, &executor);

    bool ok = (apiRet != ACL_SUCCESS);
    LOG_PRINT("[%s] Neg_PowTensorScalar_NullSelf%s\n", ok ? "PASS" : "FAIL", ok ? "" : " | expected failure but got success");

    out.Destroy();
    exponent.Destroy();
    return ok;
}

static bool NegCase_PowScalarTensor_NullExpTensor()
{
    TensorHolder out;
    ScalarHolder self;

    int ret = out.CreateZero(ACL_FLOAT, std::vector<int64_t>{2, 2});
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_PowScalarTensor_NullExpTensor | create out failed ret=%d\n", ret); return false);

    ScalarSpec selfSpec;
    selfSpec.dtype = ACL_FLOAT;
    selfSpec.value = 2.0;
    ret = self.Create(selfSpec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_PowScalarTensor_NullExpTensor | create self failed ret=%d\n", ret);
              out.Destroy(); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto apiRet = aclnnPowScalarTensorGetWorkspaceSize(self.scalar, nullptr, out.tensor, &workspaceSize, &executor);

    bool ok = (apiRet != ACL_SUCCESS);
    LOG_PRINT("[%s] Neg_PowScalarTensor_NullExpTensor%s\n", ok ? "PASS" : "FAIL", ok ? "" : " | expected failure but got success");

    out.Destroy();
    self.Destroy();
    return ok;
}

static bool NegCase_PowTensorTensor_BadOutShape()
{
    TensorSpec selfSpec;
    selfSpec.dtype = ACL_FLOAT;
    selfSpec.shape = std::vector<int64_t>{2, 3};
    selfSpec.values = MakeRamp(6, 1.0, 0.5);

    TensorSpec expSpec;
    expSpec.dtype = ACL_FLOAT;
    expSpec.shape = std::vector<int64_t>{1, 3};
    expSpec.values = std::vector<double>{1.0, 2.0, 3.0};

    TensorHolder self;
    TensorHolder exp;
    TensorHolder out;

    int ret = self.Create(selfSpec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_PowTensorTensor_BadOutShape | create self failed ret=%d\n", ret); return false);

    ret = exp.Create(expSpec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_PowTensorTensor_BadOutShape | create exp failed ret=%d\n", ret);
              self.Destroy(); return false);

    ret = out.CreateZero(ACL_FLOAT, std::vector<int64_t>{2, 2});
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_PowTensorTensor_BadOutShape | create out failed ret=%d\n", ret);
              self.Destroy(); exp.Destroy(); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto apiRet = aclnnPowTensorTensorGetWorkspaceSize(self.tensor, exp.tensor, out.tensor, &workspaceSize, &executor);

    bool ok = (apiRet != ACL_SUCCESS);
    LOG_PRINT("[%s] Neg_PowTensorTensor_BadOutShape%s\n", ok ? "PASS" : "FAIL", ok ? "" : " | expected failure but got success");

    self.Destroy();
    exp.Destroy();
    out.Destroy();
    return ok;
}

static bool NegCase_Exp2_NullSelf()
{
    TensorHolder out;
    int ret = out.CreateZero(ACL_FLOAT, std::vector<int64_t>{2, 2});
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_Exp2_NullSelf | create out failed ret=%d\n", ret); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto apiRet = aclnnExp2GetWorkspaceSize(nullptr, out.tensor, &workspaceSize, &executor);

    bool ok = (apiRet != ACL_SUCCESS);
    LOG_PRINT("[%s] Neg_Exp2_NullSelf%s\n", ok ? "PASS" : "FAIL", ok ? "" : " | expected failure but got success");

    out.Destroy();
    return ok;
}

// ---------------- suite ----------------

static bool RunAllTests(aclrtStream stream, int &passCnt, int &failCnt)
{
    bool allOk = true;

    auto RUN = [&](bool ok) {
        if (ok) {
            ++passCnt;
        } else {
            ++failCnt;
            allOk = false;
        }
    };

    // -------- TensorScalar: 特殊指数分支 --------

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2, 3};
        selfSpec.values = std::vector<double>{-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};

        ScalarSpec expScalar;
        expScalar.dtype = ACL_FLOAT;
        expScalar.value = 0.0;

        RUN(RunCase("PowTensorScalar_f32_exp0", API_POW_TENSOR_SCALAR, stream,
                    &selfSpec, nullptr, nullptr, &expScalar,
                    ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2, 3};
        selfSpec.values = std::vector<double>{-2.0, -1.0, 0.5, 1.0, 2.0, 3.0};

        ScalarSpec expScalar;
        expScalar.dtype = ACL_FLOAT;
        expScalar.value = 1.0;

        RUN(RunCase("PowTensorScalar_f32_exp1", API_POW_TENSOR_SCALAR, stream,
                    &selfSpec, nullptr, nullptr, &expScalar,
                    ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2, 4};
        selfSpec.values = std::vector<double>{0.25, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0};

        ScalarSpec expScalar;
        expScalar.dtype = ACL_FLOAT;
        expScalar.value = 0.5;

        RUN(RunCase("PowTensorScalar_f32_exp0_5_sqrt", API_POW_TENSOR_SCALAR, stream,
                    &selfSpec, nullptr, nullptr, &expScalar,
                    ACL_FLOAT, selfSpec.shape, 2e-4, 2e-4));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2, 4};
        selfSpec.values = std::vector<double>{-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0};

        ScalarSpec expScalar;
        expScalar.dtype = ACL_FLOAT;
        expScalar.value = 2.0;

        RUN(RunCase("PowTensorScalar_f32_exp2_square", API_POW_TENSOR_SCALAR, stream,
                    &selfSpec, nullptr, nullptr, &expScalar,
                    ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2, 4};
        selfSpec.values = std::vector<double>{0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 0.25, 0.125};

        ScalarSpec expScalar;
        expScalar.dtype = ACL_FLOAT;
        expScalar.value = -1.0;

        RUN(RunCase("PowTensorScalar_f32_exp_neg1_reciprocal", API_POW_TENSOR_SCALAR, stream,
                    &selfSpec, nullptr, nullptr, &expScalar,
                    ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2, 4};
        selfSpec.values = std::vector<double>{-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, -3.0};

        ScalarSpec expScalar;
        expScalar.dtype = ACL_FLOAT;
        expScalar.value = 3.0;

        RUN(RunCase("PowTensorScalar_f32_exp3_cube", API_POW_TENSOR_SCALAR, stream,
                    &selfSpec, nullptr, nullptr, &expScalar,
                    ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT16;
        selfSpec.shape = std::vector<int64_t>{8};
        selfSpec.values = std::vector<double>{0.5, 1.0, 2.0, 3.0, 4.0, 0.25, 5.0, 6.0};

        ScalarSpec expScalar;
        expScalar.dtype = ACL_FLOAT;
        expScalar.value = 2.0;

        RUN(RunCase("PowTensorScalar_f16_exp2", API_POW_TENSOR_SCALAR, stream,
                    &selfSpec, nullptr, nullptr, &expScalar,
                    ACL_FLOAT16, selfSpec.shape, 3e-2, 3e-2));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_BF16;
        selfSpec.shape = std::vector<int64_t>{8};
        selfSpec.values = std::vector<double>{0.5, 1.0, 2.0, 3.0, 4.0, 0.25, 5.0, 6.0};

        ScalarSpec expScalar;
        expScalar.dtype = ACL_FLOAT;
        expScalar.value = 3.0;

        RUN(RunCase("PowTensorScalar_bf16_exp3", API_POW_TENSOR_SCALAR, stream,
                    &selfSpec, nullptr, nullptr, &expScalar,
                    ACL_BF16, selfSpec.shape, 6e-2, 6e-2));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_INT32;
        selfSpec.shape = std::vector<int64_t>{8};
        selfSpec.values = std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8};

        ScalarSpec expScalar;
        expScalar.dtype = ACL_INT32;
        expScalar.value = 2.0;

        RUN(RunCase("PowTensorScalar_i32_exp2", API_POW_TENSOR_SCALAR, stream,
                    &selfSpec, nullptr, nullptr, &expScalar,
                    ACL_INT32, selfSpec.shape, 0.0, 0.0));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_INT8;
        selfSpec.shape = std::vector<int64_t>{8};
        selfSpec.values = std::vector<double>{1, 2, 3, 2, 1, 0, 2, 3};

        ScalarSpec expScalar;
        expScalar.dtype = ACL_INT8;
        expScalar.value = 3.0;

        RUN(RunCase("PowTensorScalar_i8_exp3", API_POW_TENSOR_SCALAR, stream,
                    &selfSpec, nullptr, nullptr, &expScalar,
                    ACL_INT8, selfSpec.shape, 0.0, 0.0));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{4};
        selfSpec.values = std::vector<double>{-4.0, -1.0, 4.0, 9.0};

        ScalarSpec expScalar;
        expScalar.dtype = ACL_FLOAT;
        expScalar.value = 0.5;

        RUN(RunCase("PowTensorScalar_f32_negative_base_fractional_exp", API_POW_TENSOR_SCALAR, stream,
                    &selfSpec, nullptr, nullptr, &expScalar,
                    ACL_FLOAT, selfSpec.shape, 2e-4, 2e-4));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{6};
        selfSpec.values = std::vector<double>{1.0, 2.0, 3.0, 4.0, 0.5, 0.25};

        ScalarSpec expScalar;
        expScalar.dtype = ACL_FLOAT;
        expScalar.value = 2.0;

        RUN(RunCase("InplacePowTensorScalar_f32_exp2", API_INPLACE_POW_TENSOR_SCALAR, stream,
                    &selfSpec, nullptr, nullptr, &expScalar,
                    ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    // -------- ScalarTensor --------

    {
        ScalarSpec selfScalar;
        selfScalar.dtype = ACL_FLOAT;
        selfScalar.value = 2.0;

        TensorSpec expSpec;
        expSpec.dtype = ACL_FLOAT;
        expSpec.shape = std::vector<int64_t>{2, 3};
        expSpec.values = std::vector<double>{-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};

        RUN(RunCase("PowScalarTensor_f32_base2", API_POW_SCALAR_TENSOR, stream,
                    nullptr, &selfScalar, &expSpec, nullptr,
                    ACL_FLOAT, expSpec.shape, 1e-5, 1e-5));
    }

    {
        ScalarSpec selfScalar;
        selfScalar.dtype = ACL_FLOAT;
        selfScalar.value = 4.0;

        TensorSpec expSpec;
        expSpec.dtype = ACL_FLOAT16;
        expSpec.shape = std::vector<int64_t>{8};
        expSpec.values = std::vector<double>{0.0, 0.5, 1.0, 2.0, 3.0, -1.0, 0.5, 2.0};

        RUN(RunCase("PowScalarTensor_base4_exp_f16", API_POW_SCALAR_TENSOR, stream,
                    nullptr, &selfScalar, &expSpec, nullptr,
                    ACL_FLOAT16, expSpec.shape, 3e-2, 3e-2));
    }

    {
        ScalarSpec selfScalar;
        selfScalar.dtype = ACL_INT32;
        selfScalar.value = 3.0;

        TensorSpec expSpec;
        expSpec.dtype = ACL_INT32;
        expSpec.shape = std::vector<int64_t>{6};
        expSpec.values = std::vector<double>{0, 1, 2, 3, 1, 2};

        RUN(RunCase("PowScalarTensor_i32_base3", API_POW_SCALAR_TENSOR, stream,
                    nullptr, &selfScalar, &expSpec, nullptr,
                    ACL_INT32, expSpec.shape, 0.0, 0.0));
    }

    // -------- TensorTensor: 7 dtype，专门打 OP_KEY --------

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_BF16;
        selfSpec.shape = std::vector<int64_t>{2, 4};
        selfSpec.values = std::vector<double>{1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5};

        TensorSpec expSpec;
        expSpec.dtype = ACL_BF16;
        expSpec.shape = std::vector<int64_t>{2, 4};
        expSpec.values = std::vector<double>{0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0};

        RUN(RunCase("PowTensorTensor_bf16", API_POW_TENSOR_TENSOR, stream,
                    &selfSpec, nullptr, &expSpec, nullptr,
                    ACL_BF16, selfSpec.shape, 6e-2, 6e-2));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT16;
        selfSpec.shape = std::vector<int64_t>{2, 4};
        selfSpec.values = std::vector<double>{1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5};

        TensorSpec expSpec;
        expSpec.dtype = ACL_FLOAT16;
        expSpec.shape = std::vector<int64_t>{2, 4};
        expSpec.values = std::vector<double>{0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0};

        RUN(RunCase("PowTensorTensor_f16", API_POW_TENSOR_TENSOR, stream,
                    &selfSpec, nullptr, &expSpec, nullptr,
                    ACL_FLOAT16, selfSpec.shape, 3e-2, 3e-2));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2, 4};
        selfSpec.values = std::vector<double>{1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5};

        TensorSpec expSpec;
        expSpec.dtype = ACL_FLOAT;
        expSpec.shape = std::vector<int64_t>{2, 4};
        expSpec.values = std::vector<double>{0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0};

        RUN(RunCase("PowTensorTensor_f32", API_POW_TENSOR_TENSOR, stream,
                    &selfSpec, nullptr, &expSpec, nullptr,
                    ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_UINT8;
        selfSpec.shape = std::vector<int64_t>{2, 4};
        selfSpec.values = std::vector<double>{1, 2, 3, 4, 2, 3, 4, 5};

        TensorSpec expSpec;
        expSpec.dtype = ACL_UINT8;
        expSpec.shape = std::vector<int64_t>{2, 4};
        expSpec.values = std::vector<double>{0, 1, 2, 1, 2, 1, 0, 1};

        RUN(RunCase("PowTensorTensor_u8", API_POW_TENSOR_TENSOR, stream,
                    &selfSpec, nullptr, &expSpec, nullptr,
                    ACL_UINT8, selfSpec.shape, 0.0, 0.0));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_INT8;
        selfSpec.shape = std::vector<int64_t>{2, 4};
        selfSpec.values = std::vector<double>{1, 2, -2, 3, -3, 2, 1, -1};

        TensorSpec expSpec;
        expSpec.dtype = ACL_INT8;
        expSpec.shape = std::vector<int64_t>{2, 4};
        expSpec.values = std::vector<double>{0, 1, 2, 3, 2, 1, 0, 3};

        RUN(RunCase("PowTensorTensor_i8", API_POW_TENSOR_TENSOR, stream,
                    &selfSpec, nullptr, &expSpec, nullptr,
                    ACL_INT8, selfSpec.shape, 0.0, 0.0));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_INT16;
        selfSpec.shape = std::vector<int64_t>{2, 4};
        selfSpec.values = std::vector<double>{1, 2, -2, 3, -3, 2, 1, -1};

        TensorSpec expSpec;
        expSpec.dtype = ACL_INT16;
        expSpec.shape = std::vector<int64_t>{2, 4};
        expSpec.values = std::vector<double>{0, 1, 2, 3, 2, 1, 0, 3};

        RUN(RunCase("PowTensorTensor_i16", API_POW_TENSOR_TENSOR, stream,
                    &selfSpec, nullptr, &expSpec, nullptr,
                    ACL_INT16, selfSpec.shape, 0.0, 0.0));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_INT32;
        selfSpec.shape = std::vector<int64_t>{2, 4};
        selfSpec.values = std::vector<double>{1, 2, -2, 3, -3, 2, 1, -1};

        TensorSpec expSpec;
        expSpec.dtype = ACL_INT32;
        expSpec.shape = std::vector<int64_t>{2, 4};
        expSpec.values = std::vector<double>{0, 1, 2, 3, 2, 1, 0, 3};

        RUN(RunCase("PowTensorTensor_i32", API_POW_TENSOR_TENSOR, stream,
                    &selfSpec, nullptr, &expSpec, nullptr,
                    ACL_INT32, selfSpec.shape, 0.0, 0.0));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2, 3, 4};
        selfSpec.values = MakeRamp(24, 1.0, 0.25);

        TensorSpec expSpec;
        expSpec.dtype = ACL_FLOAT;
        expSpec.shape = std::vector<int64_t>{1, 3, 1};
        expSpec.values = std::vector<double>{1.0, 2.0, 3.0};

        std::vector<int64_t> outShape = BroadcastShape(selfSpec.shape, expSpec.shape);
        RUN(RunCase("PowTensorTensor_f32_broadcast", API_POW_TENSOR_TENSOR, stream,
                    &selfSpec, nullptr, &expSpec, nullptr,
                    ACL_FLOAT, outShape, 1e-5, 1e-5));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2, 3};
        selfSpec.values = std::vector<double>{1.0, 2.0, 3.0, 4.0, 0.5, 0.25};

        TensorSpec expSpec;
        expSpec.dtype = ACL_FLOAT;
        expSpec.shape = std::vector<int64_t>{1, 3};
        expSpec.values = std::vector<double>{2.0, 3.0, 1.0};

        RUN(RunCase("InplacePowTensorTensor_f32_broadcast", API_INPLACE_POW_TENSOR_TENSOR, stream,
                    &selfSpec, nullptr, &expSpec, nullptr,
                    ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    // -------- Exp2 / InplaceExp2 --------

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2, 4};
        selfSpec.values = std::vector<double>{-3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0};

        RUN(RunCase("Exp2_f32", API_EXP2, stream,
                    &selfSpec, nullptr, nullptr, nullptr,
                    ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT16;
        selfSpec.shape = std::vector<int64_t>{8};
        selfSpec.values = std::vector<double>{-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 0.5, 4.0};

        RUN(RunCase("Exp2_f16", API_EXP2, stream,
                    &selfSpec, nullptr, nullptr, nullptr,
                    ACL_FLOAT16, selfSpec.shape, 3e-2, 3e-2));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{6};
        selfSpec.values = std::vector<double>{-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};

        RUN(RunCase("InplaceExp2_f32", API_INPLACE_EXP2, stream,
                    &selfSpec, nullptr, nullptr, nullptr,
                    ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    // -------- negative --------

    RUN(NegCase_PowTensorScalar_NullSelf());
    RUN(NegCase_PowScalarTensor_NullExpTensor());
    RUN(NegCase_PowTensorTensor_BadOutShape());
    RUN(NegCase_Exp2_NullSelf());

    return allOk;
}

// ---------------- main ----------------

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;

    int ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

    int passCnt = 0;
    int failCnt = 0;
    bool ok = RunAllTests(stream, passCnt, failCnt);

    LOG_PRINT("\n================ SUMMARY ================\n");
    LOG_PRINT("PASS: %d\n", passCnt);
    LOG_PRINT("FAIL: %d\n", failCnt);
    LOG_PRINT("RESULT: %s\n", ok ? "PASS" : "FAIL");
    LOG_PRINT("=========================================\n");

    Finalize(deviceId, stream);
    return ok ? 0 : 1;
}