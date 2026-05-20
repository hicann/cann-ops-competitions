/**
 * Extended end-to-end tests for aclnn Add on Ascend 910_93.
 * Covers Add / Adds / InplaceAdd / InplaceAdds / AddV3 / InplaceAddV3,
 * alpha dispatch paths, broadcasting, mixed dtype, common tiling dtype branches,
 * precision-sensitive values, and negative parameter checks.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

#define CHECK_RET(cond, expr) \
    do {                      \
        if (!(cond)) {        \
            expr;             \
        }                     \
    } while (0)

static int64_t Numel(const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return 1;
    }
    int64_t n = 1;
    for (auto v : shape) {
        n *= v;
    }
    return n;
}

static std::vector<int64_t> DefaultStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

static uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 1u : 0u));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000u) >> 13));
}

static float HalfToFloat(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

static float Bf16ToFloat(uint16_t b)
{
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

enum class DTypeKind { F32, F16, BF16, F64, I32, I64, I16, I8, U8, BOOL };

static aclDataType ToAcl(DTypeKind kind)
{
    switch (kind) {
        case DTypeKind::F32: return aclDataType::ACL_FLOAT;
        case DTypeKind::F16: return aclDataType::ACL_FLOAT16;
        case DTypeKind::BF16: return aclDataType::ACL_BF16;
        case DTypeKind::F64: return aclDataType::ACL_DOUBLE;
        case DTypeKind::I32: return aclDataType::ACL_INT32;
        case DTypeKind::I64: return aclDataType::ACL_INT64;
        case DTypeKind::I16: return aclDataType::ACL_INT16;
        case DTypeKind::I8: return aclDataType::ACL_INT8;
        case DTypeKind::U8: return aclDataType::ACL_UINT8;
        case DTypeKind::BOOL: return aclDataType::ACL_BOOL;
    }
    return aclDataType::ACL_FLOAT;
}

static size_t DTypeSize(DTypeKind kind)
{
    switch (kind) {
        case DTypeKind::F32: return sizeof(float);
        case DTypeKind::F64: return sizeof(double);
        case DTypeKind::F16:
        case DTypeKind::BF16: return sizeof(uint16_t);
        case DTypeKind::I32: return sizeof(int32_t);
        case DTypeKind::I64: return sizeof(int64_t);
        case DTypeKind::I16: return sizeof(int16_t);
        case DTypeKind::I8: return sizeof(int8_t);
        case DTypeKind::U8:
        case DTypeKind::BOOL: return sizeof(uint8_t);
    }
    return sizeof(float);
}

static std::string DTypeName(DTypeKind kind)
{
    switch (kind) {
        case DTypeKind::F32: return "float32";
        case DTypeKind::F16: return "float16";
        case DTypeKind::BF16: return "bf16";
        case DTypeKind::F64: return "double";
        case DTypeKind::I32: return "int32";
        case DTypeKind::I64: return "int64";
        case DTypeKind::I16: return "int16";
        case DTypeKind::I8: return "int8";
        case DTypeKind::U8: return "uint8";
        case DTypeKind::BOOL: return "bool";
    }
    return "unknown";
}

static std::vector<uint8_t> Encode(const std::vector<double>& values, DTypeKind kind)
{
    std::vector<uint8_t> bytes(values.size() * DTypeSize(kind));
    for (size_t i = 0; i < values.size(); ++i) {
        uint8_t* p = bytes.data() + i * DTypeSize(kind);
        switch (kind) {
            case DTypeKind::F32: {
                float v = static_cast<float>(values[i]);
                std::memcpy(p, &v, sizeof(v));
                break;
            }
            case DTypeKind::F64: {
                double v = static_cast<double>(values[i]);
                std::memcpy(p, &v, sizeof(v));
                break;
            }
            case DTypeKind::F16: {
                uint16_t v = FloatToHalf(static_cast<float>(values[i]));
                std::memcpy(p, &v, sizeof(v));
                break;
            }
            case DTypeKind::BF16: {
                uint16_t v = FloatToBf16(static_cast<float>(values[i]));
                std::memcpy(p, &v, sizeof(v));
                break;
            }
            case DTypeKind::I32: {
                int32_t v = static_cast<int32_t>(values[i]);
                std::memcpy(p, &v, sizeof(v));
                break;
            }
            case DTypeKind::I64: {
                int64_t v = static_cast<int64_t>(values[i]);
                std::memcpy(p, &v, sizeof(v));
                break;
            }
            case DTypeKind::I16: {
                int16_t v = static_cast<int16_t>(values[i]);
                std::memcpy(p, &v, sizeof(v));
                break;
            }
            case DTypeKind::I8: {
                int8_t v = static_cast<int8_t>(values[i]);
                std::memcpy(p, &v, sizeof(v));
                break;
            }
            case DTypeKind::U8: {
                uint8_t v = static_cast<uint8_t>(std::max(0.0, std::min(255.0, values[i])));
                std::memcpy(p, &v, sizeof(v));
                break;
            }
            case DTypeKind::BOOL: {
                uint8_t v = values[i] != 0.0 ? 1 : 0;
                std::memcpy(p, &v, sizeof(v));
                break;
            }
        }
    }
    return bytes;
}

static double DecodeOne(const uint8_t* p, DTypeKind kind)
{
    switch (kind) {
        case DTypeKind::F32: {
            float v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        case DTypeKind::F64: {
            double v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        case DTypeKind::F16: {
            uint16_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return HalfToFloat(v);
        }
        case DTypeKind::BF16: {
            uint16_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return Bf16ToFloat(v);
        }
        case DTypeKind::I32: {
            int32_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        case DTypeKind::I64: {
            int64_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<double>(v);
        }
        case DTypeKind::I16: {
            int16_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        case DTypeKind::I8: {
            int8_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        case DTypeKind::U8:
        case DTypeKind::BOOL: {
            uint8_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
    }
    return 0.0;
}

static std::vector<double> Decode(const std::vector<uint8_t>& bytes, DTypeKind kind)
{
    size_t n = bytes.size() / DTypeSize(kind);
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = DecodeOne(bytes.data() + i * DTypeSize(kind), kind);
    }
    return out;
}

struct ManagedTensor {
    aclTensor* tensor = nullptr;
    void* device = nullptr;
    DTypeKind dtype = DTypeKind::F32;
    std::vector<int64_t> viewShape;
    std::vector<int64_t> storageShape;
    std::vector<int64_t> strides;
    int64_t offset = 0;
    std::vector<uint8_t> hostStorage;

    ~ManagedTensor()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
        if (device != nullptr) {
            aclrtFree(device);
        }
    }

    int Create(const std::vector<double>& values, const std::vector<int64_t>& shape, DTypeKind dt)
    {
        return CreateStrided(values, shape, shape, DefaultStrides(shape), 0, dt);
    }

    int CreateStrided(const std::vector<double>& viewValues, const std::vector<int64_t>& view,
        const std::vector<int64_t>& storage, const std::vector<int64_t>& viewStrides, int64_t viewOffset, DTypeKind dt)
    {
        dtype = dt;
        viewShape = view;
        storageShape = storage;
        strides = viewStrides;
        offset = viewOffset;
        std::vector<double> storageValues(Numel(storage), -777.0);
        auto viewN = Numel(view);
        for (int64_t linear = 0; linear < viewN; ++linear) {
            int64_t t = linear;
            int64_t storageIndex = offset;
            for (int64_t d = static_cast<int64_t>(view.size()) - 1; d >= 0; --d) {
                int64_t idx = t % view[d];
                t /= view[d];
                storageIndex += idx * strides[d];
            }
            storageValues[storageIndex] = viewValues[linear];
        }
        hostStorage = Encode(storageValues, dtype);
        size_t bytes = hostStorage.size();
        if (bytes == 0) {
            hostStorage.resize(DTypeSize(dtype), 0);
            bytes = hostStorage.size();
        }
        auto ret = aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclrtMalloc failed: " << ret << std::endl; return ret);
        ret = aclrtMemcpy(device, bytes, hostStorage.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclrtMemcpy H2D failed: " << ret << std::endl; return ret);
        tensor = aclCreateTensor(viewShape.data(), viewShape.size(), ToAcl(dtype), strides.data(), offset,
            aclFormat::ACL_FORMAT_ND, storageShape.data(), storageShape.size(), device);
        CHECK_RET(tensor != nullptr, std::cout << "aclCreateTensor failed" << std::endl; return -1);
        return ACL_SUCCESS;
    }

    int CopyToHost(std::vector<uint8_t>& bytes) const
    {
        bytes.assign(hostStorage.size(), 0);
        if (bytes.empty()) {
            return ACL_SUCCESS;
        }
        auto ret = aclrtMemcpy(bytes.data(), bytes.size(), device, bytes.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclrtMemcpy D2H failed: " << ret << std::endl; return ret);
        return ACL_SUCCESS;
    }

    int CopyViewToHost(std::vector<double>& values) const
    {
        std::vector<uint8_t> bytes;
        auto ret = CopyToHost(bytes);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        int64_t viewN = Numel(viewShape);
        values.assign(viewN, 0.0);
        for (int64_t linear = 0; linear < viewN; ++linear) {
            int64_t t = linear;
            int64_t storageIndex = offset;
            for (int64_t d = static_cast<int64_t>(viewShape.size()) - 1; d >= 0; --d) {
                int64_t idx = viewShape[d] == 0 ? 0 : (t % viewShape[d]);
                if (viewShape[d] != 0) {
                    t /= viewShape[d];
                }
                storageIndex += idx * strides[d];
            }
            values[linear] = DecodeOne(bytes.data() + storageIndex * DTypeSize(dtype), dtype);
        }
        return ACL_SUCCESS;
    }
};

class ScalarHolder {
public:
    explicit ScalarHolder(double value, DTypeKind kind) : dtype_(kind), storage_(Encode({value}, kind))
    {
        scalar_ = aclCreateScalar(storage_.data(), ToAcl(kind));
    }
    ~ScalarHolder()
    {
        if (scalar_ != nullptr) {
            aclDestroyScalar(scalar_);
        }
    }
    aclScalar* Get() const { return scalar_; }
private:
    DTypeKind dtype_;
    std::vector<uint8_t> storage_;
    aclScalar* scalar_ = nullptr;
};

static int RunExecutor(uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream,
    const std::function<aclnnStatus(void*, uint64_t, aclOpExecutor*, aclrtStream)>& run)
{
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "workspace malloc failed: " << ret << std::endl; return ret);
    }
    auto ret = run(workspace, workspaceSize, executor, stream);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ret;
}

static bool NearlyEqual(double actual, double expected, DTypeKind dtype, double atol, double rtol)
{
    if (std::isnan(expected)) {
        return std::isnan(actual);
    }
    if (std::isinf(expected)) {
        return std::isinf(actual) && std::signbit(actual) == std::signbit(expected);
    }
    if (dtype == DTypeKind::I32 || dtype == DTypeKind::I64 || dtype == DTypeKind::I16 || dtype == DTypeKind::I8 || dtype == DTypeKind::U8 ||
        dtype == DTypeKind::BOOL) {
        return static_cast<int64_t>(actual) == static_cast<int64_t>(expected);
    }
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

static std::vector<int64_t> BroadcastShape(const std::vector<int64_t>& a, const std::vector<int64_t>& b)
{
    size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        int64_t ad = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
        int64_t bd = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];
        out[i] = std::max(ad, bd);
    }
    return out;
}

static std::vector<double> BroadcastBinaryExpected(const std::vector<double>& a, const std::vector<int64_t>& aShape,
    const std::vector<double>& b, const std::vector<int64_t>& bShape, double alpha, DTypeKind outType)
{
    auto outShape = BroadcastShape(aShape, bShape);
    int64_t outN = Numel(outShape);
    std::vector<double> out(outN, 0.0);
    for (int64_t linear = 0; linear < outN; ++linear) {
        int64_t t = linear;
        int64_t ai = 0;
        int64_t bi = 0;
        int64_t astride = 1;
        int64_t bstride = 1;
        for (int64_t d = static_cast<int64_t>(outShape.size()) - 1; d >= 0; --d) {
            int64_t idx = t % outShape[d];
            t /= outShape[d];
            int64_t ar = d - (static_cast<int64_t>(outShape.size()) - static_cast<int64_t>(aShape.size()));
            int64_t br = d - (static_cast<int64_t>(outShape.size()) - static_cast<int64_t>(bShape.size()));
            if (ar >= 0) {
                int64_t av = (aShape[ar] == 1) ? 0 : idx;
                ai += av * astride;
                astride *= aShape[ar];
            }
            if (br >= 0) {
                int64_t bv = (bShape[br] == 1) ? 0 : idx;
                bi += bv * bstride;
                bstride *= bShape[br];
            }
        }
        double v = a[ai] + alpha * b[bi];
        if (outType == DTypeKind::BOOL) {
            v = (v != 0.0) ? 1.0 : 0.0;
        }
        out[linear] = v;
    }
    return out;
}

static bool CheckResult(const std::string& name, const std::vector<double>& actual, const std::vector<double>& expected,
    DTypeKind dtype, double atol, double rtol)
{
    bool ok = true;
    double maxErr = 0.0;
    int bad = -1;
    for (size_t i = 0; i < expected.size(); ++i) {
        double err = std::fabs(actual[i] - expected[i]);
        if (!std::isnan(err)) {
            maxErr = std::max(maxErr, err);
        }
        if (!NearlyEqual(actual[i], expected[i], dtype, atol, rtol)) {
            ok = false;
            bad = static_cast<int>(i);
            break;
        }
    }
    std::cout << "Test: " << name << "\n  MaxError: " << std::setprecision(10) << maxErr;
    if (!ok && bad >= 0) {
        std::cout << "\n  FirstMismatch index=" << bad << " expected=" << expected[bad] << " actual=" << actual[bad];
    }
    std::cout << "\n  [" << (ok ? "PASS" : "FAIL") << "]\n";
    return ok;
}

struct Counters {
    int passed = 0;
    int failed = 0;
    void Add(bool ok)
    {
        if (ok) { ++passed; } else { ++failed; }
    }
};

static bool RunAddCase(const std::string& name, const std::vector<double>& selfData,
    const std::vector<int64_t>& selfShape, DTypeKind selfType, const std::vector<double>& otherData,
    const std::vector<int64_t>& otherShape, DTypeKind otherType, double alphaValue, DTypeKind alphaType,
    DTypeKind outType, aclrtStream stream, double atol = 1e-6, double rtol = 1e-6, bool stridedSelf = false, bool stridedOut = false)
{
    ManagedTensor self;
    int ret = ACL_SUCCESS;
    if (stridedSelf && selfShape.size() == 2) {
        std::vector<int64_t> storageShape = {selfShape[0], selfShape[1] + 1};
        std::vector<int64_t> strides = {selfShape[1] + 1, 1};
        ret = self.CreateStrided(selfData, selfShape, storageShape, strides, 0, selfType);
    } else {
        ret = self.Create(selfData, selfShape, selfType);
    }
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ManagedTensor other;
    ret = other.Create(otherData, otherShape, otherType);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    auto outShape = BroadcastShape(selfShape, otherShape);
    ManagedTensor out;
    if (stridedOut && outShape.size() == 2) {
        std::vector<int64_t> storageShape = {outShape[0], outShape[1] + 1};
        std::vector<int64_t> strides = {outShape[1] + 1, 1};
        ret = out.CreateStrided(std::vector<double>(Numel(outShape), 0.0), outShape, storageShape, strides, 0, outType);
    } else {
        ret = out.Create(std::vector<double>(Numel(outShape), 0.0), outShape, outType);
    }
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ScalarHolder alpha(alphaValue, alphaType);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.Get(), out.tensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << name << " GetWorkspace failed: " << ret << "\n  [FAIL]\n"; return false);
    ret = RunExecutor(workspaceSize, executor, stream,
        [](void* w, uint64_t s, aclOpExecutor* e, aclrtStream st) { return aclnnAdd(w, s, e, st); });
    CHECK_RET(ret == ACL_SUCCESS, std::cout << name << " Execute failed: " << ret << "\n  [FAIL]\n"; return false);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    std::vector<double> actual;
    ret = out.CopyViewToHost(actual);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    auto expected = BroadcastBinaryExpected(selfData, selfShape, otherData, otherShape, alphaValue, outType);
    return CheckResult(name + " (Add " + DTypeName(selfType) + "+" + DTypeName(otherType) + ")", actual, expected,
        outType, atol, rtol);
}


static bool RunAddWorkspaceOnlyCase(const std::string& name, const std::vector<double>& selfData,
    const std::vector<int64_t>& selfShape, DTypeKind selfType, const std::vector<double>& otherData,
    const std::vector<int64_t>& otherShape, DTypeKind otherType, double alphaValue, DTypeKind alphaType,
    DTypeKind outType, bool expectSuccess = true)
{
    ManagedTensor self;
    int ret = self.Create(selfData, selfShape, selfType);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ManagedTensor other;
    ret = other.Create(otherData, otherShape, otherType);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    auto outShape = BroadcastShape(selfShape, otherShape);
    ManagedTensor out;
    ret = out.Create(std::vector<double>(Numel(outShape), 0.0), outShape, outType);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ScalarHolder alpha(alphaValue, alphaType);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAddGetWorkspaceSize(self.tensor, other.tensor, alpha.Get(), out.tensor, &workspaceSize, &executor);
    bool ok = expectSuccess ? (ret == ACL_SUCCESS) : (ret != ACL_SUCCESS);
    std::cout << "Test: " << name << " (GetWorkspace-only) ret=" << ret
              << " workspace=" << workspaceSize << "\n  [" << (ok ? "PASS" : "FAIL") << "]\n";
    return ok;
}
static bool RunAddsCase(const std::string& name, const std::vector<double>& selfData,
    const std::vector<int64_t>& selfShape, DTypeKind selfType, double otherValue, DTypeKind otherType, double alphaValue,
    DTypeKind alphaType, DTypeKind outType, aclrtStream stream, double atol = 1e-6, double rtol = 1e-6, bool inplace = false, bool boolSpecial = false)
{
    ManagedTensor self;
    int ret = self.Create(selfData, selfShape, selfType);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ManagedTensor out;
    if (!inplace) {
        ret = out.Create(std::vector<double>(Numel(selfShape), 0.0), selfShape, outType);
        CHECK_RET(ret == ACL_SUCCESS, return false);
    }
    ScalarHolder other(otherValue, otherType);
    ScalarHolder alpha(alphaValue, alphaType);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (inplace) {
        ret = aclnnInplaceAddsGetWorkspaceSize(self.tensor, other.Get(), alpha.Get(), &workspaceSize, &executor);
    } else {
        ret = aclnnAddsGetWorkspaceSize(self.tensor, other.Get(), alpha.Get(), out.tensor, &workspaceSize, &executor);
    }
    CHECK_RET(ret == ACL_SUCCESS, std::cout << name << " GetWorkspace failed: " << ret << "\n  [FAIL]\n"; return false);
    if (inplace) {
        ret = RunExecutor(workspaceSize, executor, stream,
            [](void* w, uint64_t s, aclOpExecutor* e, aclrtStream st) { return aclnnInplaceAdds(w, s, e, st); });
    } else {
        ret = RunExecutor(workspaceSize, executor, stream,
            [](void* w, uint64_t s, aclOpExecutor* e, aclrtStream st) { return aclnnAdds(w, s, e, st); });
    }
    CHECK_RET(ret == ACL_SUCCESS, std::cout << name << " Execute failed: " << ret << "\n  [FAIL]\n"; return false);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    std::vector<double> actual;
    ret = (inplace ? self.CopyViewToHost(actual) : out.CopyViewToHost(actual));
    CHECK_RET(ret == ACL_SUCCESS, return false);
    std::vector<double> expected(selfData.size());
    for (size_t i = 0; i < selfData.size(); ++i) {
        if (boolSpecial) {
            expected[i] = (selfData[i] != 0.0 || (otherValue != 0.0 && alphaValue != 0.0)) ? 1.0 : 0.0;
        } else {
            expected[i] = selfData[i] + alphaValue * otherValue;
        }
    }
    return CheckResult(name + (inplace ? " (InplaceAdds)" : " (Adds)"), actual, expected, inplace ? selfType : outType,
        atol, rtol);
}

static bool RunInplaceAddCase(const std::string& name, const std::vector<double>& selfData,
    const std::vector<int64_t>& selfShape, DTypeKind selfType, const std::vector<double>& otherData,
    const std::vector<int64_t>& otherShape, DTypeKind otherType, double alphaValue, DTypeKind alphaType,
    aclrtStream stream, double atol = 1e-6, double rtol = 1e-6)
{
    ManagedTensor self;
    int ret = self.Create(selfData, selfShape, selfType);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ManagedTensor other;
    ret = other.Create(otherData, otherShape, otherType);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ScalarHolder alpha(alphaValue, alphaType);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnInplaceAddGetWorkspaceSize(self.tensor, other.tensor, alpha.Get(), &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << name << " GetWorkspace failed: " << ret << "\n  [FAIL]\n"; return false);
    ret = RunExecutor(workspaceSize, executor, stream,
        [](void* w, uint64_t s, aclOpExecutor* e, aclrtStream st) { return aclnnInplaceAdd(w, s, e, st); });
    CHECK_RET(ret == ACL_SUCCESS, std::cout << name << " Execute failed: " << ret << "\n  [FAIL]\n"; return false);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    std::vector<double> actual;
    ret = self.CopyViewToHost(actual);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    auto expected = BroadcastBinaryExpected(selfData, selfShape, otherData, otherShape, alphaValue, selfType);
    return CheckResult(name + " (InplaceAdd)", actual, expected, selfType, atol, rtol);
}

static bool RunAddV3Case(const std::string& name, double selfScalar, DTypeKind selfType,
    const std::vector<double>& otherData, const std::vector<int64_t>& otherShape, DTypeKind otherType, double alphaValue,
    DTypeKind alphaType, DTypeKind outType, aclrtStream stream, double atol = 1e-6, double rtol = 1e-6,
    bool inplace = false)
{
    ManagedTensor other;
    int ret = other.Create(otherData, otherShape, otherType);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    ManagedTensor out;
    if (!inplace) {
        ret = out.Create(std::vector<double>(Numel(otherShape), 0.0), otherShape, outType);
        CHECK_RET(ret == ACL_SUCCESS, return false);
    }
    ScalarHolder self(selfScalar, selfType);
    ScalarHolder alpha(alphaValue, alphaType);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (inplace) {
        ret = aclnnInplaceAddV3GetWorkspaceSize(self.Get(), other.tensor, alpha.Get(), &workspaceSize, &executor);
    } else {
        ret = aclnnAddV3GetWorkspaceSize(self.Get(), other.tensor, alpha.Get(), out.tensor, &workspaceSize, &executor);
    }
    CHECK_RET(ret == ACL_SUCCESS, std::cout << name << " GetWorkspace failed: " << ret << "\n  [FAIL]\n"; return false);
    if (inplace) {
        ret = RunExecutor(workspaceSize, executor, stream,
            [](void* w, uint64_t s, aclOpExecutor* e, aclrtStream st) { return aclnnInplaceAddV3(w, s, e, st); });
    } else {
        ret = RunExecutor(workspaceSize, executor, stream,
            [](void* w, uint64_t s, aclOpExecutor* e, aclrtStream st) { return aclnnAddV3(w, s, e, st); });
    }
    CHECK_RET(ret == ACL_SUCCESS, std::cout << name << " Execute failed: " << ret << "\n  [FAIL]\n"; return false);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    std::vector<double> actual;
    ret = (inplace ? other.CopyViewToHost(actual) : out.CopyViewToHost(actual));
    CHECK_RET(ret == ACL_SUCCESS, return false);
    std::vector<double> expected(otherData.size());
    for (size_t i = 0; i < otherData.size(); ++i) {
        expected[i] = selfScalar + alphaValue * otherData[i];
    }
    return CheckResult(name + (inplace ? " (InplaceAddV3)" : " (AddV3)"), actual, expected,
        inplace ? otherType : outType, atol, rtol);
}

static bool RunNegativeChecks(aclrtStream stream)
{
    (void)stream;
    std::cout << "Test: Negative parameter checks\n";
    bool ok = true;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    float one = 1.0f;
    ScalarHolder alpha(1.0, DTypeKind::F32);
    aclScalar* scalar = aclCreateScalar(&one, aclDataType::ACL_FLOAT);

    ManagedTensor a;
    ManagedTensor b;
    ManagedTensor out;
    ManagedTensor badOut;
    ManagedTensor rank9;
    ManagedTensor boolTensor;
    a.Create({1, 2, 3, 4, 5, 6}, {2, 3}, DTypeKind::F32);
    b.Create({1, 2, 3, 4}, {2, 2}, DTypeKind::F32);
    out.Create(std::vector<double>(6, 0), {2, 3}, DTypeKind::F32);
    badOut.Create(std::vector<double>(4, 0), {2, 2}, DTypeKind::F32);
    rank9.Create({1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, DTypeKind::F32);
    boolTensor.Create({1, 0, 1}, {3}, DTypeKind::BOOL);

    ok = ok && (aclnnAddGetWorkspaceSize(nullptr, a.tensor, alpha.Get(), out.tensor, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddGetWorkspaceSize(a.tensor, nullptr, alpha.Get(), out.tensor, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddGetWorkspaceSize(a.tensor, b.tensor, alpha.Get(), out.tensor, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddGetWorkspaceSize(a.tensor, a.tensor, nullptr, out.tensor, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddGetWorkspaceSize(a.tensor, a.tensor, alpha.Get(), nullptr, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddGetWorkspaceSize(a.tensor, a.tensor, alpha.Get(), badOut.tensor, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddGetWorkspaceSize(rank9.tensor, rank9.tensor, alpha.Get(), rank9.tensor, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddsGetWorkspaceSize(nullptr, scalar, alpha.Get(), out.tensor, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddsGetWorkspaceSize(a.tensor, nullptr, alpha.Get(), out.tensor, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddsGetWorkspaceSize(a.tensor, scalar, nullptr, out.tensor, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddsGetWorkspaceSize(a.tensor, scalar, alpha.Get(), nullptr, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddsGetWorkspaceSize(a.tensor, scalar, alpha.Get(), badOut.tensor, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddV3GetWorkspaceSize(nullptr, a.tensor, alpha.Get(), out.tensor, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddV3GetWorkspaceSize(scalar, nullptr, alpha.Get(), out.tensor, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddV3GetWorkspaceSize(scalar, a.tensor, nullptr, out.tensor, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddV3GetWorkspaceSize(scalar, a.tensor, alpha.Get(), nullptr, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnAddV3GetWorkspaceSize(scalar, boolTensor.tensor, alpha.Get(), boolTensor.tensor, &workspaceSize, &executor) != ACL_SUCCESS);
    ok = ok && (aclnnInplaceAddGetWorkspaceSize(b.tensor, a.tensor, alpha.Get(), &workspaceSize, &executor) != ACL_SUCCESS);
    aclDestroyScalar(scalar);

    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "]\n";
    return ok;
}

static int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclInit failed: " << ret << std::endl; return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclrtSetDevice failed: " << ret << std::endl; return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "aclrtCreateStream failed: " << ret << std::endl; return ret);
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    Counters counters;
    auto add = [&](bool ok) { counters.Add(ok); };

    add(RunAddCase("01 basic fp32 alpha=1 direct Add", {0, 1, 2, 3, 4, 5}, {2, 3}, DTypeKind::F32,
        {1, 1, 1, 2, 2, 2}, {2, 3}, DTypeKind::F32, 1.0, DTypeKind::F32, DTypeKind::F32, stream));
    add(RunAddCase("02 fp32 broadcast alpha=2.5 Axpy", {0, 1, 2, 3, 4, 5}, {2, 3}, DTypeKind::F32,
        {10, 20, 30}, {1, 3}, DTypeKind::F32, 2.5, DTypeKind::F32, DTypeKind::F32, stream));
    add(RunAddCase("03 fp32 strided self non-contiguous", {1, 2, 3, 4}, {2, 2}, DTypeKind::F32,
        {5, 6, 7, 8}, {2, 2}, DTypeKind::F32, 1.0, DTypeKind::F32, DTypeKind::F32, stream, 1e-6, 1e-6, true));
    add(RunAddCase("03b fp32 strided output ViewCopy", {1, 2, 3, 4}, {2, 2}, DTypeKind::F32,
        {5, 6, 7, 8}, {2, 2}, DTypeKind::F32, 1.0, DTypeKind::F32, DTypeKind::F32, stream, 1e-6, 1e-6, false, true));
    add(RunAddCase("04b mixed fp32 + fp16 alpha=2 cast+Axpy", {1.5, -2.0, 3.25, 4.5}, {4}, DTypeKind::F32,
        {0.5, 2.0, -3.0, 5.0}, {4}, DTypeKind::F16, 2.0, DTypeKind::F32, DTypeKind::F32, stream, 1e-2, 1e-2));
    add(RunAddCase("04 mixed fp16 + fp32 alpha=1 mixed-kernel", {1.5, -2.0, 3.25, 4.5}, {4}, DTypeKind::F16,
        {0.5, 2.0, -3.0, 5.0}, {4}, DTypeKind::F32, 1.0, DTypeKind::F32, DTypeKind::F32, stream, 1e-2, 1e-2));
    add(RunAddCase("05 fp16 alpha=-1.5", {1.0, 2.0, -3.0, 4.0}, {4}, DTypeKind::F16,
        {0.5, -1.0, 2.0, -4.0}, {4}, DTypeKind::F16, -1.5, DTypeKind::F32, DTypeKind::F16, stream, 2e-2, 2e-2));
    add(RunAddCase("06 bf16 alpha=1 tiling", {1.25, -2.5, 3.5, 4.25}, {4}, DTypeKind::BF16,
        {2.0, 3.0, -4.0, 5.0}, {4}, DTypeKind::BF16, 1.0, DTypeKind::F32, DTypeKind::BF16, stream, 5e-2, 5e-2));
    add(RunAddCase("07 int32 alpha=-2", {10, -20, 30, -40}, {4}, DTypeKind::I32,
        {1, 2, 3, 4}, {4}, DTypeKind::I32, -2, DTypeKind::I32, DTypeKind::I32, stream));
    add(RunAddCase("08 int64 alpha=1", {10000000000.0, -7.0}, {2}, DTypeKind::I64,
        {3.0, 5.0}, {2}, DTypeKind::I64, 1, DTypeKind::I64, DTypeKind::I64, stream));
    add(RunAddCase("09 int8 alpha=1", {10, -20, 30, -40}, {4}, DTypeKind::I8,
        {1, 2, -3, 4}, {4}, DTypeKind::I8, 1, DTypeKind::I8, DTypeKind::I8, stream));
    add(RunAddCase("10 uint8 alpha=1", {10, 20, 30, 40}, {4}, DTypeKind::U8,
        {1, 2, 3, 4}, {4}, DTypeKind::U8, 1, DTypeKind::U8, DTypeKind::U8, stream));
    add(RunAddCase("11 bool alpha=true", {1, 0, 1, 0}, {4}, DTypeKind::BOOL,
        {0, 0, 1, 1}, {4}, DTypeKind::BOOL, 1, DTypeKind::BOOL, DTypeKind::BOOL, stream));
    add(RunAddCase("06b bf16 alpha=2 Axpy", {1.25, -2.5, 3.5, 4.25}, {4}, DTypeKind::BF16,
        {2.0, 3.0, -4.0, 5.0}, {4}, DTypeKind::BF16, 2.0, DTypeKind::F32, DTypeKind::BF16, stream, 8e-2, 8e-2));
    add(RunAddCase("07b int64 alpha=2 AxpyV2", {100, -200, 300}, {3}, DTypeKind::I64,
        {7, 8, -9}, {3}, DTypeKind::I64, 2, DTypeKind::I64, DTypeKind::I64, stream));
    add(RunAddCase("09b int8 alpha=-1 AxpyV2", {10, -20, 30, -40}, {4}, DTypeKind::I8,
        {1, 2, -3, 4}, {4}, DTypeKind::I8, -1, DTypeKind::I8, DTypeKind::I8, stream));
    add(RunAddCase("11b empty tensor Add early return", {}, {0}, DTypeKind::F32,
        {}, {0}, DTypeKind::F32, 1.0, DTypeKind::F32, DTypeKind::F32, stream));
    add(RunAddCase("11c mixed bf16 + fp32 alpha=1 mixed-kernel", {1.5, -2.0, 3.25}, {3}, DTypeKind::BF16,
        {0.5, 2.0, -3.0}, {3}, DTypeKind::F32, 1.0, DTypeKind::F32, DTypeKind::F32, stream, 5e-2, 5e-2));
    add(RunAddCase("12 precision large + small fp32", {1e10, 1e10}, {2}, DTypeKind::F32,
        {1e-5, -1e-5}, {2}, DTypeKind::F32, 1.0, DTypeKind::F32, DTypeKind::F32, stream, 1e-6, 1e-6));
    add(RunAddCase("13 precision cancellation fp32", {1.0000001, 2.0000001}, {2}, DTypeKind::F32,
        {-1.0, -2.0}, {2}, DTypeKind::F32, 1.0, DTypeKind::F32, DTypeKind::F32, stream, 1e-6, 1e-6));

    add(RunAddsCase("14 Adds fp32 scalar alpha=-3", {1, 2, 3, 4}, {4}, DTypeKind::F32,
        0.25, DTypeKind::F32, -3.0, DTypeKind::F32, DTypeKind::F32, stream));
    add(RunAddsCase("15 Adds int32 scalar alpha=3", {10, 20, -30}, {3}, DTypeKind::I32,
        4, DTypeKind::I32, 3, DTypeKind::I32, DTypeKind::I32, stream));
    add(RunAddsCase("16 InplaceAdds fp32 scalar alpha=0", {7, 8, 9}, {3}, DTypeKind::F32,
        100, DTypeKind::F32, 0, DTypeKind::F32, DTypeKind::F32, stream, 1e-6, 1e-6, true));

    add(RunAddsCase("16b Adds bool special cast to int32", {1, 0, 1, 0}, {4}, DTypeKind::BOOL,
        1, DTypeKind::BOOL, 1, DTypeKind::BOOL, DTypeKind::I32, stream, 0, 0, false, true));
    add(RunAddsCase("16c Adds empty tensor early return", {}, {0}, DTypeKind::F32,
        1.0, DTypeKind::F32, 1.0, DTypeKind::F32, DTypeKind::F32, stream));
    add(RunAddsCase("16d Adds bf16 exact scalar keep-bf16", {1.0, -2.0, 3.0}, {3}, DTypeKind::BF16,
        1.0, DTypeKind::F32, 1.0, DTypeKind::F32, DTypeKind::BF16, stream, 8e-2, 8e-2));
    add(RunAddsCase("16e Adds bf16 inexact scalar promote-float", {1.0, -2.0, 3.0}, {3}, DTypeKind::BF16,
        0.1, DTypeKind::F32, 1.3, DTypeKind::F32, DTypeKind::F32, stream, 8e-2, 8e-2));
    add(RunInplaceAddCase("17 InplaceAdd fp32 broadcast other", {1, 2, 3, 4, 5, 6}, {2, 3}, DTypeKind::F32,
        {10, 20, 30}, {1, 3}, DTypeKind::F32, 1.5, DTypeKind::F32, stream));
    add(RunInplaceAddCase("18 InplaceAdd int32 alpha=1", {1, 2, 3}, {3}, DTypeKind::I32,
        {4, 5, 6}, {3}, DTypeKind::I32, 1, DTypeKind::I32, stream));

    add(RunAddV3Case("19 AddV3 scalar self alpha=1", 10.0, DTypeKind::F32,
        {1, 2, 3, 4}, {4}, DTypeKind::F32, 1.0, DTypeKind::F32, DTypeKind::F32, stream));
    add(RunAddV3Case("20 AddV3 scalar self alpha=2.25 Axpy", -3.0, DTypeKind::F32,
        {1, -2, 3, -4}, {4}, DTypeKind::F32, 2.25, DTypeKind::F32, DTypeKind::F32, stream));
    add(RunAddV3Case("21 AddV3 int32 Mul+Add path", 5, DTypeKind::I32,
        {1, 2, 3}, {3}, DTypeKind::I32, -2, DTypeKind::I32, DTypeKind::I32, stream));
    add(RunAddV3Case("21b AddV3 int8 alpha=2 Mul+Add fallback", 2, DTypeKind::I32,
        {1, -2, 3}, {3}, DTypeKind::I8, 2, DTypeKind::I8, DTypeKind::I8, stream));
    add(RunAddV3Case("22 InplaceAddV3 fp32", 2.0, DTypeKind::F32,
        {1, 2, 3}, {3}, DTypeKind::F32, 3.0, DTypeKind::F32, DTypeKind::F32, stream, 1e-6, 1e-6, true));

    add(RunAddV3Case("22b AddV3 bf16 alpha=2 Mul+Add fallback", 2.0, DTypeKind::F32,
        {1.0, -2.0, 3.0}, {3}, DTypeKind::BF16, 2.0, DTypeKind::F32, DTypeKind::BF16, stream, 8e-2, 8e-2));
    add(RunAddV3Case("22c AddV3 empty tensor early return", 2.0, DTypeKind::F32,
        {}, {0}, DTypeKind::F32, 1.0, DTypeKind::F32, DTypeKind::F32, stream));
    add(RunAddWorkspaceOnlyCase("22d double alpha=2 fallback/AiCpu dispatch", {1.0, -2.0}, {2}, DTypeKind::F64,
        {0.25, 0.5}, {2}, DTypeKind::F64, 2.0, DTypeKind::F64, DTypeKind::F64));
    add(RunAddWorkspaceOnlyCase("22e int16 alpha=1 fallback/AiCpu dispatch", {1, -2, 3}, {3}, DTypeKind::I16,
        {4, 5, -6}, {3}, DTypeKind::I16, 1.0, DTypeKind::I16, DTypeKind::I16));
    add(RunNegativeChecks(stream));

    std::cout << "Summary: " << counters.passed << " passed, " << counters.failed << " failed" << std::endl;

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return counters.failed == 0 ? 0 : 1;
}
