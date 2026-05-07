/**
 * test_aclnn_add.cpp
 *
 * 设计目标：
 * 1. 严格按官方骨架风格写：正式 include + 正式 aclTensor / aclScalar / aclOpExecutor
 * 2. 覆盖 Add / Adds / InplaceAdd / InplaceAdds / AddV3 / InplaceAddV3
 * 3. 做真实执行 + CPU 端期望值校验
 * 4. 覆盖 float32 / float16 / bf16 / int32 / mixed dtype / broadcast / alpha 分支 / error path
 *
 * 说明：
 * - 如果你本地分支的 InplaceAddV3 精确签名与题面不同，只改 RunInplaceAddV3 包装函数即可。
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

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

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
    int64_t size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        size *= shape[i];
    }
    return size;
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

static std::vector<double> ExpectedTensorTensor(
    const TensorSpec &selfSpec,
    const TensorSpec &otherSpec,
    aclDataType outDtype,
    const std::vector<int64_t> &outShape,
    double alpha)
{
    std::vector<uint8_t> selfBytes = PackTensor(selfSpec);
    std::vector<uint8_t> otherBytes = PackTensor(otherSpec);

    int64_t outNumel = GetShapeSize(outShape);
    std::vector<double> expected(static_cast<size_t>(outNumel), 0.0);

    for (int64_t i = 0; i < outNumel; ++i) {
        std::vector<int64_t> coord = UnravelIndex(i, outShape);
        int64_t aoff = BroadcastOffset(coord, selfSpec.shape);
        int64_t boff = BroadcastOffset(coord, otherSpec.shape);

        double a = ReadValueAsDouble(selfBytes, selfSpec.dtype, aoff);
        double b = ReadValueAsDouble(otherBytes, otherSpec.dtype, boff);
        expected[static_cast<size_t>(i)] = CastLikeDtype(outDtype, a + alpha * b);
    }
    return expected;
}

static std::vector<double> ExpectedTensorScalar(
    const TensorSpec &selfSpec,
    const ScalarSpec &otherScalar,
    aclDataType outDtype,
    const std::vector<int64_t> &outShape,
    double alpha)
{
    std::vector<uint8_t> selfBytes = PackTensor(selfSpec);
    double b = CastLikeDtype(otherScalar.dtype, otherScalar.value);

    int64_t outNumel = GetShapeSize(outShape);
    std::vector<double> expected(static_cast<size_t>(outNumel), 0.0);

    for (int64_t i = 0; i < outNumel; ++i) {
        std::vector<int64_t> coord = UnravelIndex(i, outShape);
        int64_t aoff = BroadcastOffset(coord, selfSpec.shape);
        double a = ReadValueAsDouble(selfBytes, selfSpec.dtype, aoff);
        expected[static_cast<size_t>(i)] = CastLikeDtype(outDtype, a + alpha * b);
    }
    return expected;
}

static std::vector<double> ExpectedScalarTensor(
    const ScalarSpec &selfScalar,
    const TensorSpec &otherSpec,
    aclDataType outDtype,
    const std::vector<int64_t> &outShape,
    double alpha)
{
    double a = CastLikeDtype(selfScalar.dtype, selfScalar.value);
    std::vector<uint8_t> otherBytes = PackTensor(otherSpec);

    int64_t outNumel = GetShapeSize(outShape);
    std::vector<double> expected(static_cast<size_t>(outNumel), 0.0);

    for (int64_t i = 0; i < outNumel; ++i) {
        std::vector<int64_t> coord = UnravelIndex(i, outShape);
        int64_t boff = BroadcastOffset(coord, otherSpec.shape);
        double b = ReadValueAsDouble(otherBytes, otherSpec.dtype, boff);
        expected[static_cast<size_t>(i)] = CastLikeDtype(outDtype, a + alpha * b);
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ret=%d\n", ret); return ret);

    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ret=%d\n", ret); return ret);

    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ret=%d\n", ret); return ret);

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

static int RunAdd(
    aclTensor *self, aclTensor *other, aclScalar *alpha, aclTensor *out, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    ret = aclrtSynchronizeStream(stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

static int RunAdds(
    aclTensor *self, aclScalar *other, aclScalar *alpha, aclTensor *out, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    auto ret = aclnnAddsGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    ret = aclrtSynchronizeStream(stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

static int RunInplaceAdd(
    aclTensor *selfRef, aclTensor *other, aclScalar *alpha, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    auto ret = aclnnInplaceAddGetWorkspaceSize(selfRef, other, alpha, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    ret = aclrtSynchronizeStream(stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

static int RunInplaceAdds(
    aclTensor *selfRef, aclScalar *other, aclScalar *alpha, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    auto ret = aclnnInplaceAddsGetWorkspaceSize(selfRef, other, alpha, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    ret = aclrtSynchronizeStream(stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

static int RunAddV3(
    aclScalar *self, aclTensor *other, aclScalar *alpha, aclTensor *out, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    auto ret = aclnnAddV3GetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnAddV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    ret = aclrtSynchronizeStream(stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

/*
 * 这里按题面语义采用：
 *   selfRef(tensor) += alpha * other(scalar)
 * 如果你本地 aclnn_add_v3.h 的 InplaceAddV3 精确签名不同，只改这个函数即可。
 */
static int RunInplaceAddV3(
    aclScalar *selfRef, aclTensor *other, aclScalar *alpha, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    auto ret = aclnnInplaceAddV3GetWorkspaceSize(selfRef, other, alpha, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnInplaceAddV3(workspaceAddr, workspaceSize, executor, stream);
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
    API_ADD = 0,
    API_ADDS = 1,
    API_INPLACE_ADD = 2,
    API_INPLACE_ADDS = 3,
    API_ADD_V3 = 4,
    API_INPLACE_ADD_V3 = 5
};

static bool RunCase(
    const std::string &name,
    ApiKind apiKind,
    aclrtStream stream,
    const TensorSpec *selfTensorSpec,
    const ScalarSpec *selfScalarSpec,
    const TensorSpec *otherTensorSpec,
    const ScalarSpec *otherScalarSpec,
    const ScalarSpec &alphaSpec,
    aclDataType outDtype,
    const std::vector<int64_t> &outShape,
    double atol,
    double rtol)
{
    TensorHolder selfTensor;
    TensorHolder otherTensor;
    TensorHolder outTensor;
    ScalarHolder selfScalar;
    ScalarHolder otherScalar;
    ScalarHolder alpha;

    int ret = ACL_SUCCESS;

    if (selfTensorSpec != nullptr) {
        ret = selfTensor.Create(*selfTensorSpec);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s | create self tensor failed ret=%d\n", name.c_str(), ret); return false);
    }

    if (otherTensorSpec != nullptr) {
        ret = otherTensor.Create(*otherTensorSpec);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s | create other tensor failed ret=%d\n", name.c_str(), ret); selfTensor.Destroy(); return false);
    }

    if (selfScalarSpec != nullptr) {
        ret = selfScalar.Create(*selfScalarSpec);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s | create self scalar failed ret=%d\n", name.c_str(), ret);
                  selfTensor.Destroy(); otherTensor.Destroy(); return false);
    }

    if (otherScalarSpec != nullptr) {
        ret = otherScalar.Create(*otherScalarSpec);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s | create other scalar failed ret=%d\n", name.c_str(), ret);
                  selfTensor.Destroy(); otherTensor.Destroy(); selfScalar.Destroy(); return false);
    }

    ret = alpha.Create(alphaSpec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s | create alpha failed ret=%d\n", name.c_str(), ret);
              selfTensor.Destroy(); otherTensor.Destroy(); selfScalar.Destroy(); otherScalar.Destroy(); return false);

    if (apiKind == API_ADD || apiKind == API_ADDS || apiKind == API_ADD_V3) {
        ret = outTensor.CreateZero(outDtype, outShape);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] %s | create out failed ret=%d\n", name.c_str(), ret);
                  selfTensor.Destroy(); otherTensor.Destroy(); selfScalar.Destroy(); otherScalar.Destroy(); alpha.Destroy(); return false);
    }

    switch (apiKind) {
        case API_ADD:
            ret = RunAdd(selfTensor.tensor, otherTensor.tensor, alpha.scalar, outTensor.tensor, stream);
            break;
        case API_ADDS:
            ret = RunAdds(selfTensor.tensor, otherScalar.scalar, alpha.scalar, outTensor.tensor, stream);
            break;
        case API_INPLACE_ADD:
            ret = RunInplaceAdd(selfTensor.tensor, otherTensor.tensor, alpha.scalar, stream);
            break;
        case API_INPLACE_ADDS:
            ret = RunInplaceAdds(selfTensor.tensor, otherScalar.scalar, alpha.scalar, stream);
            break;
        case API_ADD_V3:
            ret = RunAddV3(selfScalar.scalar, otherTensor.tensor, alpha.scalar, outTensor.tensor, stream);
            break;
        case API_INPLACE_ADD_V3:
            ret = RunInplaceAddV3(selfScalar.scalar, otherTensor.tensor, alpha.scalar, stream);
            break;
        default:
            ret = -9999;
            break;
    }

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s | execute failed ret=%d\n", name.c_str(), ret);
        selfTensor.Destroy();
        otherTensor.Destroy();
        outTensor.Destroy();
        selfScalar.Destroy();
        otherScalar.Destroy();
        alpha.Destroy();
        return false;
    }

    std::vector<double> expected;
    TensorHolder *actual = nullptr;

    switch (apiKind) {
        case API_ADD:
            expected = ExpectedTensorTensor(*selfTensorSpec, *otherTensorSpec, outDtype, outShape, alphaSpec.value);
            actual = &outTensor;
            break;
        case API_ADDS:
            expected = ExpectedTensorScalar(*selfTensorSpec, *otherScalarSpec, outDtype, outShape, alphaSpec.value);
            actual = &outTensor;
            break;
        case API_INPLACE_ADD:
            expected = ExpectedTensorTensor(*selfTensorSpec, *otherTensorSpec, selfTensorSpec->dtype, selfTensorSpec->shape, alphaSpec.value);
            actual = &selfTensor;
            break;
        case API_INPLACE_ADDS:
            expected = ExpectedTensorScalar(*selfTensorSpec, *otherScalarSpec, selfTensorSpec->dtype, selfTensorSpec->shape, alphaSpec.value);
            actual = &selfTensor;
            break;
        case API_ADD_V3:
            expected = ExpectedScalarTensor(*selfScalarSpec, *otherTensorSpec, outDtype, outShape, alphaSpec.value);
            actual = &outTensor;
            break;
        case API_INPLACE_ADD_V3:
            expected = ExpectedScalarTensor(*selfScalarSpec, *otherTensorSpec,
                                otherTensorSpec->dtype,
                                otherTensorSpec->shape,
                                alphaSpec.value);
            actual = &otherTensor;
            break;
        default:
            break;
    }

    ret = actual->CopyBack();
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FAIL] %s | copy back failed ret=%d\n", name.c_str(), ret);
        selfTensor.Destroy();
        otherTensor.Destroy();
        outTensor.Destroy();
        selfScalar.Destroy();
        otherScalar.Destroy();
        alpha.Destroy();
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
    otherTensor.Destroy();
    outTensor.Destroy();
    selfScalar.Destroy();
    otherScalar.Destroy();
    alpha.Destroy();
    return ok;
}

// ---------------- negative cases ----------------

static bool NegCase_Add_NullSelf()
{
    TensorSpec otherSpec;
    otherSpec.dtype = ACL_FLOAT;
    otherSpec.shape = std::vector<int64_t>{2, 3};
    otherSpec.values = MakeRamp(6, 1.0, 1.0);

    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    int ret = other.Create(otherSpec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_Add_NullSelf | create other failed ret=%d\n", ret); return false);

    ret = out.CreateZero(ACL_FLOAT, std::vector<int64_t>{2, 3});
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_Add_NullSelf | create out failed ret=%d\n", ret); other.Destroy(); return false);

    ScalarSpec alphaSpec;
    alphaSpec.dtype = ACL_FLOAT;
    alphaSpec.value = 1.0;
    ret = alpha.Create(alphaSpec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_Add_NullSelf | create alpha failed ret=%d\n", ret);
              other.Destroy(); out.Destroy(); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto apiRet = aclnnAddGetWorkspaceSize(nullptr, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);

    bool ok = (apiRet != ACL_SUCCESS);
    LOG_PRINT("[%s] Neg_Add_NullSelf%s\n", ok ? "PASS" : "FAIL", ok ? "" : " | expected failure but got success");

    other.Destroy();
    out.Destroy();
    alpha.Destroy();
    return ok;
}

static bool NegCase_Add_InvalidMix()
{
    TensorSpec selfSpec;
    selfSpec.dtype = ACL_BOOL;
    selfSpec.shape = std::vector<int64_t>{2, 2};
    selfSpec.values = std::vector<double>{1, 0, 1, 0};

    TensorSpec otherSpec;
    otherSpec.dtype = ACL_FLOAT;
    otherSpec.shape = std::vector<int64_t>{2, 2};
    otherSpec.values = std::vector<double>{1.0, 2.0, 3.0, 4.0};

    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    int ret = self.Create(selfSpec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_Add_InvalidMix | create self failed ret=%d\n", ret); return false);

    ret = other.Create(otherSpec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_Add_InvalidMix | create other failed ret=%d\n", ret);
              self.Destroy(); return false);

    ret = out.CreateZero(ACL_FLOAT, std::vector<int64_t>{2, 2});
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_Add_InvalidMix | create out failed ret=%d\n", ret);
              self.Destroy(); other.Destroy(); return false);

    ScalarSpec alphaSpec;
    alphaSpec.dtype = ACL_FLOAT;
    alphaSpec.value = 1.0;
    ret = alpha.Create(alphaSpec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_Add_InvalidMix | create alpha failed ret=%d\n", ret);
              self.Destroy(); other.Destroy(); out.Destroy(); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto apiRet = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);

    bool ok = (apiRet != ACL_SUCCESS);
    LOG_PRINT("[%s] Neg_Add_InvalidMix%s\n", ok ? "PASS" : "FAIL", ok ? "" : " | expected failure but got success");

    self.Destroy();
    other.Destroy();
    out.Destroy();
    alpha.Destroy();
    return ok;
}

static bool NegCase_Add_BadOutShape()
{
    TensorSpec selfSpec;
    selfSpec.dtype = ACL_FLOAT;
    selfSpec.shape = std::vector<int64_t>{2, 3};
    selfSpec.values = MakeRamp(6, 0.0, 1.0);

    TensorSpec otherSpec;
    otherSpec.dtype = ACL_FLOAT;
    otherSpec.shape = std::vector<int64_t>{1, 3};
    otherSpec.values = std::vector<double>{10.0, 20.0, 30.0};

    TensorHolder self;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    int ret = self.Create(selfSpec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_Add_BadOutShape | create self failed ret=%d\n", ret); return false);

    ret = other.Create(otherSpec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_Add_BadOutShape | create other failed ret=%d\n", ret);
              self.Destroy(); return false);

    // 故意给错 out shape，正确广播应为 [2, 3]
    ret = out.CreateZero(ACL_FLOAT, std::vector<int64_t>{2, 2});
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_Add_BadOutShape | create out failed ret=%d\n", ret);
              self.Destroy(); other.Destroy(); return false);

    ScalarSpec alphaSpec;
    alphaSpec.dtype = ACL_FLOAT;
    alphaSpec.value = 1.0;
    ret = alpha.Create(alphaSpec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_Add_BadOutShape | create alpha failed ret=%d\n", ret);
              self.Destroy(); other.Destroy(); out.Destroy(); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto apiRet = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);

    bool ok = (apiRet != ACL_SUCCESS);
    LOG_PRINT("[%s] Neg_Add_BadOutShape%s\n", ok ? "PASS" : "FAIL", ok ? "" : " | expected failure but got success");

    self.Destroy();
    other.Destroy();
    out.Destroy();
    alpha.Destroy();
    return ok;
}

static bool NegCase_AddV3_NullSelfScalar()
{
    TensorSpec otherSpec;
    otherSpec.dtype = ACL_FLOAT;
    otherSpec.shape = std::vector<int64_t>{4};
    otherSpec.values = std::vector<double>{1.0, 2.0, 3.0, 4.0};

    TensorHolder other;
    TensorHolder out;
    ScalarHolder alpha;

    int ret = other.Create(otherSpec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_AddV3_NullSelfScalar | create other failed ret=%d\n", ret); return false);

    ret = out.CreateZero(ACL_FLOAT, std::vector<int64_t>{4});
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_AddV3_NullSelfScalar | create out failed ret=%d\n", ret);
              other.Destroy(); return false);

    ScalarSpec alphaSpec;
    alphaSpec.dtype = ACL_FLOAT;
    alphaSpec.value = 1.0;
    ret = alpha.Create(alphaSpec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[FAIL] Neg_AddV3_NullSelfScalar | create alpha failed ret=%d\n", ret);
              other.Destroy(); out.Destroy(); return false);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto apiRet = aclnnAddV3GetWorkspaceSize(nullptr, other.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);

    bool ok = (apiRet != ACL_SUCCESS);
    LOG_PRINT("[%s] Neg_AddV3_NullSelfScalar%s\n", ok ? "PASS" : "FAIL", ok ? "" : " | expected failure but got success");

    other.Destroy();
    out.Destroy();
    alpha.Destroy();
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

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{4, 2};
        selfSpec.values = std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7};

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_FLOAT;
        otherSpec.shape = std::vector<int64_t>{4, 2};
        otherSpec.values = std::vector<double>{1, 1, 1, 2, 2, 2, 3, 3};

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = 1.0;

        RUN(RunCase("Add_f32_same_alpha1", API_ADD, stream,
                    &selfSpec, nullptr, &otherSpec, nullptr,
                    alpha, ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2, 3};
        selfSpec.values = std::vector<double>{-2.5, 0.0, 1.0, 4.0, 7.5, -9.0};

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_FLOAT;
        otherSpec.shape = std::vector<int64_t>{2, 3};
        otherSpec.values = std::vector<double>{10, 20, 30, 40, 50, 60};

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = 0.0;

        RUN(RunCase("Add_f32_same_alpha0", API_ADD, stream,
                    &selfSpec, nullptr, &otherSpec, nullptr,
                    alpha, ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2, 1, 3, 1, 4};
        selfSpec.values = MakeRamp(GetShapeSize(selfSpec.shape), -1.0, 0.25);

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_FLOAT;
        otherSpec.shape = std::vector<int64_t>{1, 5, 1, 6, 1};
        otherSpec.values = MakeRamp(GetShapeSize(otherSpec.shape), 2.0, -0.1);

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = -0.75;

        std::vector<int64_t> outShape = BroadcastShape(selfSpec.shape, otherSpec.shape);
        RUN(RunCase("Add_f32_rank5_broadcast_alpha_neg", API_ADD, stream,
                    &selfSpec, nullptr, &otherSpec, nullptr,
                    alpha, ACL_FLOAT, outShape, 1e-5, 1e-5));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2048};
        selfSpec.values = MakeRamp(2048, -3.0, 0.01);

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_FLOAT;
        otherSpec.shape = std::vector<int64_t>{2048};
        otherSpec.values = MakeRamp(2048, 5.0, -0.02);

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = 1.0;

        RUN(RunCase("Add_f32_large_tensor", API_ADD, stream,
                    &selfSpec, nullptr, &otherSpec, nullptr,
                    alpha, ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT16;
        selfSpec.shape = std::vector<int64_t>{2, 8};
        selfSpec.values = MakeRamp(16, -1.25, 0.125);

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_FLOAT16;
        otherSpec.shape = std::vector<int64_t>{2, 8};
        otherSpec.values = MakeRamp(16, 0.5, -0.0625);

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = 1.5;

        RUN(RunCase("Add_f16_same_alpha1_5", API_ADD, stream,
                    &selfSpec, nullptr, &otherSpec, nullptr,
                    alpha, ACL_FLOAT16, selfSpec.shape, 3e-2, 3e-2));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_BF16;
        selfSpec.shape = std::vector<int64_t>{3, 5};
        selfSpec.values = MakeRamp(15, -2.0, 0.2);

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_BF16;
        otherSpec.shape = std::vector<int64_t>{3, 5};
        otherSpec.values = MakeRamp(15, 3.0, -0.15);

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = -0.5;

        RUN(RunCase("Add_bf16_same_alpha_neg0_5", API_ADD, stream,
                    &selfSpec, nullptr, &otherSpec, nullptr,
                    alpha, ACL_BF16, selfSpec.shape, 6e-2, 6e-2));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT16;
        selfSpec.shape = std::vector<int64_t>{2, 3, 4};
        selfSpec.values = MakeRamp(24, -1.0, 0.1);

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_FLOAT;
        otherSpec.shape = std::vector<int64_t>{2, 3, 4};
        otherSpec.values = MakeRamp(24, 2.0, -0.05);

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = 1.0;

        RUN(RunCase("Add_f16_f32_to_f32", API_ADD, stream,
                    &selfSpec, nullptr, &otherSpec, nullptr,
                    alpha, ACL_FLOAT, selfSpec.shape, 1e-4, 1e-4));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2, 3, 4};
        selfSpec.values = MakeRamp(24, 1.0, 0.03);

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_FLOAT16;
        otherSpec.shape = std::vector<int64_t>{2, 3, 4};
        otherSpec.values = MakeRamp(24, -2.0, 0.07);

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = 0.75;

        RUN(RunCase("Add_f32_f16_to_f32", API_ADD, stream,
                    &selfSpec, nullptr, &otherSpec, nullptr,
                    alpha, ACL_FLOAT, selfSpec.shape, 1e-4, 1e-4));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_BF16;
        selfSpec.shape = std::vector<int64_t>{4, 4};
        selfSpec.values = MakeRamp(16, -0.8, 0.11);

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_FLOAT;
        otherSpec.shape = std::vector<int64_t>{4, 4};
        otherSpec.values = MakeRamp(16, 1.2, -0.04);

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = 1.0;

        RUN(RunCase("Add_bf16_f32_to_f32", API_ADD, stream,
                    &selfSpec, nullptr, &otherSpec, nullptr,
                    alpha, ACL_FLOAT, selfSpec.shape, 1e-4, 1e-4));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{4, 4};
        selfSpec.values = MakeRamp(16, 0.6, 0.09);

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_BF16;
        otherSpec.shape = std::vector<int64_t>{4, 4};
        otherSpec.values = MakeRamp(16, -1.5, 0.05);

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = -2.0;

        RUN(RunCase("Add_f32_bf16_to_f32", API_ADD, stream,
                    &selfSpec, nullptr, &otherSpec, nullptr,
                    alpha, ACL_FLOAT, selfSpec.shape, 1e-4, 1e-4));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_INT32;
        selfSpec.shape = std::vector<int64_t>{3, 4};
        selfSpec.values = std::vector<double>{1, -2, 3, -4, 5, -6, 7, -8, 9, 10, -11, 12};

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_INT32;
        otherSpec.shape = std::vector<int64_t>{3, 4};
        otherSpec.values = std::vector<double>{2, 3, -4, 5, -6, 7, 8, -9, 10, -11, 12, 13};

        ScalarSpec alpha;
        alpha.dtype = ACL_INT32;
        alpha.value = 2.0;

        RUN(RunCase("Add_i32_same_alpha2", API_ADD, stream,
                    &selfSpec, nullptr, &otherSpec, nullptr,
                    alpha, ACL_INT32, selfSpec.shape, 0.0, 0.0));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2, 3, 4};
        selfSpec.values = MakeRamp(24, -4.0, 0.5);

        ScalarSpec otherScalar;
        otherScalar.dtype = ACL_FLOAT;
        otherScalar.value = 2.25;

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = -2.0;

        RUN(RunCase("Adds_f32_scalar_alpha_neg2", API_ADDS, stream,
                    &selfSpec, nullptr, nullptr, &otherScalar,
                    alpha, ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_INT32;
        selfSpec.shape = std::vector<int64_t>{8};
        selfSpec.values = std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7};

        ScalarSpec otherScalar;
        otherScalar.dtype = ACL_INT32;
        otherScalar.value = -3.0;

        ScalarSpec alpha;
        alpha.dtype = ACL_INT32;
        alpha.value = 3.0;

        RUN(RunCase("Adds_i32_scalar_alpha3", API_ADDS, stream,
                    &selfSpec, nullptr, nullptr, &otherScalar,
                    alpha, ACL_INT32, selfSpec.shape, 0.0, 0.0));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{2, 3, 4};
        selfSpec.values = MakeRamp(24, 0.0, 1.0);

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_FLOAT;
        otherSpec.shape = std::vector<int64_t>{1, 3, 1};
        otherSpec.values = std::vector<double>{1.0, -2.0, 3.0};

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = 1.25;

        RUN(RunCase("InplaceAdd_f32_broadcast", API_INPLACE_ADD, stream,
                    &selfSpec, nullptr, &otherSpec, nullptr,
                    alpha, ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{5, 5};
        selfSpec.values = MakeRamp(25, -5.0, 0.4);

        ScalarSpec otherScalar;
        otherScalar.dtype = ACL_FLOAT;
        otherScalar.value = -1.5;

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = 2.5;

        RUN(RunCase("InplaceAdds_f32_scalar", API_INPLACE_ADDS, stream,
                    &selfSpec, nullptr, nullptr, &otherScalar,
                    alpha, ACL_FLOAT, selfSpec.shape, 1e-5, 1e-5));
    }

    {
        ScalarSpec selfScalar;
        selfScalar.dtype = ACL_FLOAT;
        selfScalar.value = 3.5;

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_FLOAT;
        otherSpec.shape = std::vector<int64_t>{2, 3, 4};
        otherSpec.values = MakeRamp(24, -2.0, 0.2);

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = 1.0;

        RUN(RunCase("AddV3_scalar_f32_tensor_alpha1", API_ADD_V3, stream,
                    nullptr, &selfScalar, &otherSpec, nullptr,
                    alpha, ACL_FLOAT, otherSpec.shape, 1e-5, 1e-5));
    }

    {
        ScalarSpec selfScalar;
        selfScalar.dtype = ACL_FLOAT;
        selfScalar.value = -1.25;

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_FLOAT;
        otherSpec.shape = std::vector<int64_t>{3, 7};
        otherSpec.values = MakeRamp(21, 4.0, -0.3);

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = -0.5;

        RUN(RunCase("AddV3_scalar_f32_tensor_alpha_neg0_5", API_ADD_V3, stream,
                    nullptr, &selfScalar, &otherSpec, nullptr,
                    alpha, ACL_FLOAT, otherSpec.shape, 1e-5, 1e-5));
    }

    {
        ScalarSpec selfScalar;
        selfScalar.dtype = ACL_INT32;
        selfScalar.value = 7.0;

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_INT32;
        otherSpec.shape = std::vector<int64_t>{6};
        otherSpec.values = std::vector<double>{1, -2, 3, -4, 5, -6};

        ScalarSpec alpha;
        alpha.dtype = ACL_INT32;
        alpha.value = 2.0;

        RUN(RunCase("AddV3_scalar_i32_tensor_alpha2", API_ADD_V3, stream,
                    nullptr, &selfScalar, &otherSpec, nullptr,
                    alpha, ACL_INT32, otherSpec.shape, 0.0, 0.0));
    }

    {
        TensorSpec selfSpec;
        selfSpec.dtype = ACL_FLOAT;
        selfSpec.shape = std::vector<int64_t>{10};
        selfSpec.values = MakeRamp(10, 1.0, 0.25);

        ScalarSpec selfScalar;
        selfScalar.dtype = ACL_FLOAT;
        selfScalar.value = 4.0;

        TensorSpec otherSpec;
        otherSpec.dtype = ACL_FLOAT;
        otherSpec.shape = std::vector<int64_t>{10};
        otherSpec.values = MakeRamp(10, 1.0, 0.25);

        ScalarSpec alpha;
        alpha.dtype = ACL_FLOAT;
        alpha.value = -1.5;

        RUN(RunCase("InplaceAddV3_scalar_path", API_INPLACE_ADD_V3, stream,
                    nullptr, &selfScalar, &otherSpec, nullptr,
                    alpha, ACL_FLOAT, otherSpec.shape, 1e-5, 1e-5));
    }

    RUN(NegCase_Add_NullSelf());
    RUN(NegCase_Add_InvalidMix());
    RUN(NegCase_Add_BadOutShape());
    RUN(NegCase_AddV3_NullSelfScalar());

    return allOk;
}

// ---------------- main ----------------

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;

    int ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ret=%d\n", ret); return ret);

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