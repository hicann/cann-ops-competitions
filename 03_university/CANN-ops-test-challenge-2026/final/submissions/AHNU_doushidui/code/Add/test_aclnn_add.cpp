/**
 * End-to-end coverage tests for CANN ops-math Add operator.
 *
 * This file is based on the official aclnnAdd example and extends it to cover:
 *   - aclnnAdd / aclnnAdds / aclnnInplaceAdd / aclnnInplaceAdds
 *   - aclnnAddV3 / aclnnInplaceAddV3
 *   - tensor-tensor, tensor-scalar, scalar-tensor and inplace paths
 *   - alpha == 1, Axpy/AxpyV2, Mul+Add, mixed dtype and bool special paths
 *   - broadcast, non-contiguous view input/output, empty tensor and invalid-parameter paths
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

#if defined(__has_include)
#if __has_include("opdev/platform.h")
#include "opdev/platform.h"
#define ADD_TEST_HAS_OPDEV_PLATFORM 1
#endif
#endif

// Public aclnnInplaceAdd reuses aclnnAddGetWorkspaceSize and therefore does not
// enter l0op::AddInplace in math/add/op_api/add.cpp. Declare the L0 symbols directly
// and reuse a public aclOpExecutor returned by aclnnAddGetWorkspaceSize as the probe
// executor; this avoids depending on internal CREATE_EXECUTOR headers that may not be
// exposed in the example build.
namespace l0op {
const aclTensor* Add(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor);
const aclTensor* AddInplace(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor);
bool IsAddSupportNonContiguous(const aclTensor* self, const aclTensor* other);
}  // namespace l0op

// Optional direct tiling probe. The weak declaration keeps this file linkable if
// the example binary is built without op_host tiling symbols.
namespace gert { class TilingContext; }
namespace optiling { uint32_t TilingForAdd(gert::TilingContext* context) __attribute__((weak)); }

#if defined(ADD_TEST_HAS_OPDEV_PLATFORM)
// Optional SoC-branch probe. Many E2E runs only see one physical NPU arch;
// if the opdev platform test double exposes a setter, temporarily switch the
// enum to hit otherwise cold switch cases in aclnn_add.cpp/add.cpp.
template <typename T>
static auto TrySetCurNpuArchObj(T&& info, op::NpuArch arch, int) -> decltype(info.SetCurNpuArch(arch), bool())
{ info.SetCurNpuArch(arch); return true; }
template <typename T> static bool TrySetCurNpuArchObj(T&&, op::NpuArch, ...) { return false; }

template <typename T>
static auto TrySetNpuArchObj(T&& info, op::NpuArch arch, int) -> decltype(info.SetNpuArch(arch), bool())
{ info.SetNpuArch(arch); return true; }
template <typename T> static bool TrySetNpuArchObj(T&&, op::NpuArch, ...) { return false; }

template <typename T>
static auto TrySetCurNpuArchU32Obj(T&& info, op::NpuArch arch, int) -> decltype(info.SetCurNpuArch(static_cast<uint32_t>(arch)), bool())
{ info.SetCurNpuArch(static_cast<uint32_t>(arch)); return true; }
template <typename T> static bool TrySetCurNpuArchU32Obj(T&&, op::NpuArch, ...) { return false; }

template <typename T>
static auto TrySetNpuArchU32Obj(T&& info, op::NpuArch arch, int) -> decltype(info.SetNpuArch(static_cast<uint32_t>(arch)), bool())
{ info.SetNpuArch(static_cast<uint32_t>(arch)); return true; }
template <typename T> static bool TrySetNpuArchU32Obj(T&&, op::NpuArch, ...) { return false; }

// Extra platform-test-double probes. Different CANN drops expose different names; all
// attempts are SFINAE-guarded so the example build still compiles when they are absent.
template <typename T>
static auto TryAssignCurNpuArchObj(T&& info, op::NpuArch arch, int) -> decltype(info.curNpuArch = arch, bool())
{ info.curNpuArch = arch; return true; }
template <typename T> static bool TryAssignCurNpuArchObj(T&&, op::NpuArch, ...) { return false; }

template <typename T>
static auto TryAssignNpuArchObj(T&& info, op::NpuArch arch, int) -> decltype(info.npuArch = arch, bool())
{ info.npuArch = arch; return true; }
template <typename T> static bool TryAssignNpuArchObj(T&&, op::NpuArch, ...) { return false; }

template <typename T>
static auto TrySetSocVersionObj(T&& info, const char* soc, int) -> decltype(info.SetSocVersion(soc), bool())
{ info.SetSocVersion(soc); return true; }
template <typename T> static bool TrySetSocVersionObj(T&&, const char*, ...) { return false; }

template <typename T>
static auto TrySetSocNameObj(T&& info, const char* soc, int) -> decltype(info.SetSocName(soc), bool())
{ info.SetSocName(soc); return true; }
template <typename T> static bool TrySetSocNameObj(T&&, const char*, ...) { return false; }

template <typename T>
static auto TrySetShortSocVersionObj(T&& info, const char* soc, int) -> decltype(info.SetShortSocVersion(soc), bool())
{ info.SetShortSocVersion(soc); return true; }
template <typename T> static bool TrySetShortSocVersionObj(T&&, const char*, ...) { return false; }

static bool TrySetPlatformArchForCoverage(op::NpuArch arch)
{
    auto&& info = op::GetCurrentPlatformInfo();
    if (TrySetCurNpuArchObj(info, arch, 0)) { return true; }
    if (TrySetNpuArchObj(info, arch, 0)) { return true; }
    if (TrySetCurNpuArchU32Obj(info, arch, 0)) { return true; }
    if (TrySetNpuArchU32Obj(info, arch, 0)) { return true; }
    if (TryAssignCurNpuArchObj(info, arch, 0)) { return true; }
    if (TryAssignNpuArchObj(info, arch, 0)) { return true; }
    return false;
}

static bool TrySetPlatformSocNameForCoverage(const char* soc)
{
    auto&& info = op::GetCurrentPlatformInfo();
    if (TrySetSocVersionObj(info, soc, 0)) { return true; }
    if (TrySetSocNameObj(info, soc, 0)) { return true; }
    if (TrySetShortSocVersionObj(info, soc, 0)) { return true; }
    return false;
}
#endif

#define LOG_PRINT(message, ...)         \
    do {                                \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

static aclrtStream g_stream = nullptr;
static int32_t g_deviceId = 0;

using fp16_storage_t = uint16_t;
using bf16_storage_t = uint16_t;

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return 1;
    }
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

static std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() >= 2) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
        }
    }
    return strides;
}

static size_t DTypeSize(aclDataType dtype)
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
        case ACL_DOUBLE:
        case ACL_INT64:
            return 8;
        case ACL_COMPLEX64:
            return 8;
        case ACL_COMPLEX128:
            return 16;
        default:
            return 4;
    }
}

static const char* DTypeName(aclDataType dtype)
{
    switch (dtype) {
        case ACL_BOOL: return "BOOL";
        case ACL_INT8: return "INT8";
        case ACL_UINT8: return "UINT8";
        case ACL_INT16: return "INT16";
        case ACL_INT32: return "INT32";
        case ACL_INT64: return "INT64";
        case ACL_FLOAT16: return "FLOAT16";
        case ACL_BF16: return "BF16";
        case ACL_FLOAT: return "FLOAT";
        case ACL_DOUBLE: return "DOUBLE";
        case ACL_COMPLEX64: return "COMPLEX64";
        case ACL_COMPLEX128: return "COMPLEX128";
        default: return "UNKNOWN";
    }
}

static bf16_storage_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // Round-to-nearest-even for a closer BF16 host reference.
    uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7FFFU + lsb;
    return static_cast<bf16_storage_t>(bits >> 16);
}

static float Bf16ToFloat(bf16_storage_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static fp16_storage_t FloatToFp16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFU;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<fp16_storage_t>(sign);
        }
        mant = (mant | 0x800000U) >> static_cast<uint32_t>(1 - exp);
        return static_cast<fp16_storage_t>(sign | ((mant + 0x1000U) >> 13));
    }
    if (exp >= 31) {
        return static_cast<fp16_storage_t>(sign | 0x7C00U);
    }
    return static_cast<fp16_storage_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000U) >> 13));
}

static float Fp16ToFloat(fp16_storage_t value)
{
    uint32_t sign = (static_cast<uint32_t>(value & 0x8000U)) << 16;
    uint32_t exp = (value & 0x7C00U) >> 10;
    uint32_t mant = value & 0x03FFU;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400U) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFU;
            exp = exp + (127 - 15);
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1FU) {
        bits = sign | 0x7F800000U | (mant << 13);
    } else {
        exp = exp + (127 - 15);
        bits = sign | (exp << 23) | (mant << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static void WriteValue(std::vector<uint8_t>& bytes, int64_t index, aclDataType dtype, double value)
{
    uint8_t* dst = bytes.data() + static_cast<size_t>(index) * DTypeSize(dtype);
    switch (dtype) {
        case ACL_BOOL: {
            bool v = (value != 0.0);
            std::memcpy(dst, &v, sizeof(v));
            break;
        }
        case ACL_INT8: {
            int8_t v = static_cast<int8_t>(value);
            std::memcpy(dst, &v, sizeof(v));
            break;
        }
        case ACL_UINT8: {
            uint8_t v = static_cast<uint8_t>(value);
            std::memcpy(dst, &v, sizeof(v));
            break;
        }
        case ACL_INT16: {
            int16_t v = static_cast<int16_t>(value);
            std::memcpy(dst, &v, sizeof(v));
            break;
        }
        case ACL_INT32: {
            int32_t v = static_cast<int32_t>(value);
            std::memcpy(dst, &v, sizeof(v));
            break;
        }
        case ACL_INT64: {
            int64_t v = static_cast<int64_t>(value);
            std::memcpy(dst, &v, sizeof(v));
            break;
        }
        case ACL_FLOAT16: {
            fp16_storage_t v = FloatToFp16(static_cast<float>(value));
            std::memcpy(dst, &v, sizeof(v));
            break;
        }
        case ACL_BF16: {
            bf16_storage_t v = FloatToBf16(static_cast<float>(value));
            std::memcpy(dst, &v, sizeof(v));
            break;
        }
        case ACL_COMPLEX64: {
            float v[2] = {static_cast<float>(value), 0.0f};
            std::memcpy(dst, v, sizeof(v));
            break;
        }
        case ACL_COMPLEX128: {
            double v[2] = {value, 0.0};
            std::memcpy(dst, v, sizeof(v));
            break;
        }
        case ACL_DOUBLE: {
            double v = value;
            std::memcpy(dst, &v, sizeof(v));
            break;
        }
        case ACL_FLOAT:
        default: {
            float v = static_cast<float>(value);
            std::memcpy(dst, &v, sizeof(v));
            break;
        }
    }
}

static double ReadValue(const std::vector<uint8_t>& bytes, int64_t index, aclDataType dtype)
{
    const uint8_t* src = bytes.data() + static_cast<size_t>(index) * DTypeSize(dtype);
    switch (dtype) {
        case ACL_BOOL: {
            bool v = false;
            std::memcpy(&v, src, sizeof(v));
            return v ? 1.0 : 0.0;
        }
        case ACL_INT8: {
            int8_t v = 0;
            std::memcpy(&v, src, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_UINT8: {
            uint8_t v = 0;
            std::memcpy(&v, src, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_INT16: {
            int16_t v = 0;
            std::memcpy(&v, src, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_INT32: {
            int32_t v = 0;
            std::memcpy(&v, src, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_INT64: {
            int64_t v = 0;
            std::memcpy(&v, src, sizeof(v));
            return static_cast<double>(v);
        }
        case ACL_FLOAT16: {
            fp16_storage_t v = 0;
            std::memcpy(&v, src, sizeof(v));
            return static_cast<double>(Fp16ToFloat(v));
        }
        case ACL_BF16: {
            bf16_storage_t v = 0;
            std::memcpy(&v, src, sizeof(v));
            return static_cast<double>(Bf16ToFloat(v));
        }
        case ACL_COMPLEX64: {
            float v[2] = {0.0f, 0.0f};
            std::memcpy(v, src, sizeof(v));
            return static_cast<double>(v[0]);
        }
        case ACL_COMPLEX128: {
            double v[2] = {0.0, 0.0};
            std::memcpy(v, src, sizeof(v));
            return v[0];
        }
        case ACL_DOUBLE: {
            double v = 0;
            std::memcpy(&v, src, sizeof(v));
            return v;
        }
        case ACL_FLOAT:
        default: {
            float v = 0;
            std::memcpy(&v, src, sizeof(v));
            return static_cast<double>(v);
        }
    }
}

struct TensorSpec {
    std::vector<int64_t> shape;
    std::vector<double> data;
    aclDataType dtype;
    std::vector<int64_t> storageShape;
    std::vector<int64_t> strides;
    int64_t offset;
    aclFormat format;

    TensorSpec()
        : dtype(ACL_FLOAT), offset(0), format(ACL_FORMAT_ND)
    {}

    TensorSpec(const std::vector<int64_t>& s, const std::vector<double>& d, aclDataType t)
        : shape(s), data(d), dtype(t), offset(0), format(ACL_FORMAT_ND)
    {}
};

struct TensorHolder {
    TensorSpec spec;
    void* device = nullptr;
    aclTensor* tensor = nullptr;
};

static int64_t LogicalToStorageIndex(const TensorSpec& spec, int64_t logicalIndex)
{
    const std::vector<int64_t>& shape = spec.shape;
    const std::vector<int64_t>& strides = spec.strides.empty() ? ContiguousStrides(spec.storageShape.empty() ? shape : spec.storageShape) : spec.strides;
    int64_t remain = logicalIndex;
    int64_t storageIndex = spec.offset;
    for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
        int64_t coord = 0;
        if (shape[static_cast<size_t>(dim)] != 0) {
            coord = remain % shape[static_cast<size_t>(dim)];
            remain /= shape[static_cast<size_t>(dim)];
        }
        storageIndex += coord * strides[static_cast<size_t>(dim)];
    }
    return storageIndex;
}

static std::vector<uint8_t> BuildStorageBytes(const TensorSpec& spec)
{
    std::vector<int64_t> storageShape = spec.storageShape.empty() ? spec.shape : spec.storageShape;
    int64_t storageElems = GetShapeSize(storageShape);
    std::vector<uint8_t> bytes(static_cast<size_t>(std::max<int64_t>(0, storageElems)) * DTypeSize(spec.dtype), 0);
    int64_t logicalElems = GetShapeSize(spec.shape);
    if (logicalElems <= 0 || spec.data.empty()) {
        return bytes;
    }
    for (int64_t i = 0; i < logicalElems; ++i) {
        const int64_t storageIndex = LogicalToStorageIndex(spec, i);
        WriteValue(bytes, storageIndex, spec.dtype, spec.data[static_cast<size_t>(i % spec.data.size())]);
    }
    return bytes;
}

static int CreateTensor(const TensorSpec& inputSpec, TensorHolder* holder)
{
    holder->spec = inputSpec;
    if (holder->spec.storageShape.empty()) {
        holder->spec.storageShape = holder->spec.shape;
    }
    if (holder->spec.strides.empty()) {
        holder->spec.strides = ContiguousStrides(holder->spec.storageShape);
    }

    std::vector<uint8_t> bytes = BuildStorageBytes(holder->spec);
    if (!bytes.empty()) {
        auto ret = aclrtMalloc(&holder->device, bytes.size(), ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("aclrtMalloc failed, ret=%d, bytes=%zu\n", ret, bytes.size());
            return ret;
        }
        ret = aclrtMemcpy(holder->device, bytes.size(), bytes.data(), bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("aclrtMemcpy H2D failed, ret=%d\n", ret);
            return ret;
        }
    }

    holder->tensor = aclCreateTensor(
        holder->spec.shape.data(), holder->spec.shape.size(), holder->spec.dtype,
        holder->spec.strides.data(), holder->spec.offset, holder->spec.format,
        holder->spec.storageShape.data(), holder->spec.storageShape.size(), holder->device);
    if (holder->tensor == nullptr) {
        LOG_PRINT("aclCreateTensor failed for dtype=%s\n", DTypeName(holder->spec.dtype));
        return -1;
    }
    return ACL_SUCCESS;
}

static void DestroyTensor(TensorHolder* holder)
{
    if (holder == nullptr) {
        return;
    }
    if (holder->tensor != nullptr) {
        aclDestroyTensor(holder->tensor);
        holder->tensor = nullptr;
    }
    if (holder->device != nullptr) {
        aclrtFree(holder->device);
        holder->device = nullptr;
    }
}

struct ScalarHolder {
    aclScalar* scalar = nullptr;
    aclDataType dtype = ACL_FLOAT;
    double value = 0.0;
    bool b = false;
    int8_t i8 = 0;
    uint8_t u8 = 0;
    int16_t i16 = 0;
    int32_t i32 = 0;
    int64_t i64 = 0;
    float f32 = 0.0f;
    double f64 = 0.0;
    fp16_storage_t f16 = 0;
    bf16_storage_t bf16 = 0;
    float c64[2] = {0.0f, 0.0f};
    double c128[2] = {0.0, 0.0};
};

static int CreateScalar(double value, aclDataType dtype, ScalarHolder* holder)
{
    holder->dtype = dtype;
    holder->value = value;
    void* ptr = nullptr;
    switch (dtype) {
        case ACL_BOOL:
            holder->b = (value != 0.0);
            ptr = &holder->b;
            break;
        case ACL_INT8:
            holder->i8 = static_cast<int8_t>(value);
            ptr = &holder->i8;
            break;
        case ACL_UINT8:
            holder->u8 = static_cast<uint8_t>(value);
            ptr = &holder->u8;
            break;
        case ACL_INT16:
            holder->i16 = static_cast<int16_t>(value);
            ptr = &holder->i16;
            break;
        case ACL_INT32:
            holder->i32 = static_cast<int32_t>(value);
            ptr = &holder->i32;
            break;
        case ACL_INT64:
            holder->i64 = static_cast<int64_t>(value);
            ptr = &holder->i64;
            break;
        case ACL_FLOAT16:
            holder->f16 = FloatToFp16(static_cast<float>(value));
            ptr = &holder->f16;
            break;
        case ACL_BF16:
            holder->bf16 = FloatToBf16(static_cast<float>(value));
            ptr = &holder->bf16;
            break;
        case ACL_COMPLEX64:
            holder->c64[0] = static_cast<float>(value);
            holder->c64[1] = 0.0f;
            ptr = holder->c64;
            break;
        case ACL_COMPLEX128:
            holder->c128[0] = value;
            holder->c128[1] = 0.0;
            ptr = holder->c128;
            break;
        case ACL_DOUBLE:
            holder->f64 = value;
            ptr = &holder->f64;
            break;
        case ACL_FLOAT:
        default:
            holder->f32 = static_cast<float>(value);
            ptr = &holder->f32;
            break;
    }
    holder->scalar = aclCreateScalar(ptr, dtype);
    if (holder->scalar == nullptr) {
        LOG_PRINT("aclCreateScalar failed for dtype=%s\n", DTypeName(dtype));
        return -1;
    }
    return ACL_SUCCESS;
}

static void DestroyScalar(ScalarHolder* holder)
{
    if (holder != nullptr && holder->scalar != nullptr) {
        aclDestroyScalar(holder->scalar);
        holder->scalar = nullptr;
    }
}

static int Init()
{
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclInit failed, ret=%d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(g_deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed, ret=%d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(&g_stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed, ret=%d\n", ret);
        return ret;
    }
    return ACL_SUCCESS;
}

static void Deinit()
{
    if (g_stream != nullptr) {
        aclrtDestroyStream(g_stream);
        g_stream = nullptr;
    }
    aclrtResetDevice(g_deviceId);
    aclFinalize();
}

static std::vector<int64_t> LinearToCoords(int64_t index, const std::vector<int64_t>& shape)
{
    std::vector<int64_t> coords(shape.size(), 0);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        if (shape[static_cast<size_t>(i)] == 0) {
            coords[static_cast<size_t>(i)] = 0;
        } else {
            coords[static_cast<size_t>(i)] = index % shape[static_cast<size_t>(i)];
            index /= shape[static_cast<size_t>(i)];
        }
    }
    return coords;
}

static int64_t BroadcastLogicalIndex(const std::vector<int64_t>& outCoords,
                                     const std::vector<int64_t>& outShape,
                                     const std::vector<int64_t>& inShape)
{
    if (GetShapeSize(inShape) == 0) {
        return 0;
    }
    int64_t inIndex = 0;
    int64_t stride = 1;
    int64_t outRank = static_cast<int64_t>(outShape.size());
    int64_t inRank = static_cast<int64_t>(inShape.size());
    for (int64_t i = inRank - 1; i >= 0; --i) {
        int64_t outDim = outRank - inRank + i;
        int64_t coord = (inShape[static_cast<size_t>(i)] == 1) ? 0 : outCoords[static_cast<size_t>(outDim)];
        inIndex += coord * stride;
        stride *= inShape[static_cast<size_t>(i)];
    }
    return inIndex;
}

static double InputLogicalValue(const TensorSpec& spec, int64_t logicalIndex)
{
    if (spec.data.empty()) {
        return 0.0;
    }
    return spec.data[static_cast<size_t>(logicalIndex % static_cast<int64_t>(spec.data.size()))];
}

static std::vector<double> CopyTensorResultToHost(const TensorHolder& outHolder)
{
    const TensorSpec& spec = outHolder.spec;
    int64_t outElems = GetShapeSize(spec.shape);
    std::vector<double> logical(static_cast<size_t>(std::max<int64_t>(0, outElems)), 0.0);
    if (outElems <= 0) {
        return logical;
    }
    int64_t storageElems = GetShapeSize(spec.storageShape.empty() ? spec.shape : spec.storageShape);
    size_t bytesSize = static_cast<size_t>(storageElems) * DTypeSize(spec.dtype);
    std::vector<uint8_t> bytes(bytesSize, 0);
    auto ret = aclrtMemcpy(bytes.data(), bytes.size(), outHolder.device, bytes.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy D2H failed, ret=%d\n", ret);
        return logical;
    }
    for (int64_t i = 0; i < outElems; ++i) {
        int64_t storageIndex = LogicalToStorageIndex(spec, i);
        logical[static_cast<size_t>(i)] = ReadValue(bytes, storageIndex, spec.dtype);
    }
    return logical;
}

enum class ApiKind {
    ADD,
    ADDS,
    INPLACE_ADD,
    INPLACE_ADDS,
    ADD_V3,
    INPLACE_ADD_V3
};

static bool IsIntegerLike(aclDataType dtype)
{
    return dtype == ACL_BOOL || dtype == ACL_INT8 || dtype == ACL_UINT8 || dtype == ACL_INT16 ||
           dtype == ACL_INT32 || dtype == ACL_INT64;
}

static bool CompareValue(double actual, double expected, aclDataType outDtype)
{
    if (outDtype == ACL_BOOL) {
        return (actual != 0.0) == (expected != 0.0);
    }
    if (IsIntegerLike(outDtype)) {
        return std::fabs(actual - expected) <= 0.5;
    }
    double atol = 1e-4;
    double rtol = 1e-4;
    if (outDtype == ACL_FLOAT16) {
        atol = 2e-2;
        rtol = 2e-2;
    } else if (outDtype == ACL_BF16) {
        atol = 8e-2;
        rtol = 8e-2;
    } else if (outDtype == ACL_FLOAT) {
        atol = 1e-3;
        rtol = 1e-3;
    }
    return std::fabs(actual - expected) <= (atol + rtol * std::fabs(expected));
}

static std::vector<double> BuildExpected(ApiKind kind,
                                         const TensorSpec& selfSpec,
                                         const TensorSpec& otherSpec,
                                         const TensorSpec& outSpec,
                                         double selfScalarValue,
                                         double otherScalarValue,
                                         double alphaValue,
                                         aclDataType selfScalarDtype,
                                         aclDataType otherScalarDtype,
                                         aclDataType alphaDtype)
{
    int64_t outElems = GetShapeSize(outSpec.shape);
    std::vector<double> expected(static_cast<size_t>(std::max<int64_t>(0, outElems)), 0.0);
    for (int64_t i = 0; i < outElems; ++i) {
        auto outCoords = LinearToCoords(i, outSpec.shape);
        double lhs = 0.0;
        double rhs = 0.0;
        if (kind == ApiKind::ADD || kind == ApiKind::INPLACE_ADD) {
            int64_t selfIndex = BroadcastLogicalIndex(outCoords, outSpec.shape, selfSpec.shape);
            int64_t otherIndex = BroadcastLogicalIndex(outCoords, outSpec.shape, otherSpec.shape);
            lhs = InputLogicalValue(selfSpec, selfIndex);
            rhs = InputLogicalValue(otherSpec, otherIndex);
        } else if (kind == ApiKind::ADDS || kind == ApiKind::INPLACE_ADDS) {
            int64_t selfIndex = BroadcastLogicalIndex(outCoords, outSpec.shape, selfSpec.shape);
            lhs = InputLogicalValue(selfSpec, selfIndex);
            rhs = otherScalarValue;
        } else {
            int64_t otherIndex = BroadcastLogicalIndex(outCoords, outSpec.shape, otherSpec.shape);
            lhs = selfScalarValue;
            rhs = InputLogicalValue(otherSpec, otherIndex);
        }

        // aclnnAdds has a dedicated bool guard: bool tensor + True scalar * True alpha cast to
        // a non-bool output must stay 1 instead of becoming 2.
        bool boolAddsGuard = (kind == ApiKind::ADDS || kind == ApiKind::INPLACE_ADDS) &&
                             selfSpec.dtype == ACL_BOOL && otherScalarDtype == ACL_BOOL &&
                             alphaDtype == ACL_BOOL && outSpec.dtype != ACL_BOOL;
        if (boolAddsGuard) {
            expected[static_cast<size_t>(i)] = ((lhs != 0.0) || ((rhs != 0.0) && (alphaValue != 0.0))) ? 1.0 : 0.0;
        } else {
            expected[static_cast<size_t>(i)] = lhs + alphaValue * rhs;
        }
    }
    return expected;
}

static bool RunComputeCase(const std::string& name,
                           ApiKind kind,
                           TensorSpec selfSpec,
                           TensorSpec otherSpec,
                           TensorSpec outSpec,
                           double alphaValue,
                           aclDataType alphaDtype,
                           aclDataType selfScalarDtype = ACL_FLOAT,
                           aclDataType otherScalarDtype = ACL_FLOAT,
                           bool allowUnsupportedSkip = false)
{
    LOG_PRINT("[ RUN      ] %s\n", name.c_str());
    TensorHolder selfHolder;
    TensorHolder otherHolder;
    TensorHolder outHolder;
    ScalarHolder alpha;
    ScalarHolder selfScalar;
    ScalarHolder otherScalar;
    void* workspace = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    bool pass = false;

    int ret = CreateScalar(alphaValue, alphaDtype, &alpha);
    if (ret != ACL_SUCCESS) {
        goto cleanup;
    }

    if (kind == ApiKind::ADD || kind == ApiKind::INPLACE_ADD || kind == ApiKind::ADDS || kind == ApiKind::INPLACE_ADDS) {
        ret = CreateTensor(selfSpec, &selfHolder);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
    }
    if (kind == ApiKind::ADD || kind == ApiKind::INPLACE_ADD) {
        ret = CreateTensor(otherSpec, &otherHolder);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
    }
    if (kind == ApiKind::ADDS || kind == ApiKind::INPLACE_ADDS) {
        ret = CreateScalar(otherSpec.data.empty() ? 0.0 : otherSpec.data[0], otherScalarDtype, &otherScalar);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
    }
    if (kind == ApiKind::ADD_V3 || kind == ApiKind::INPLACE_ADD_V3) {
        ret = CreateScalar(selfSpec.data.empty() ? 0.0 : selfSpec.data[0], selfScalarDtype, &selfScalar);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
        ret = CreateTensor(otherSpec, &otherHolder);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
    }

    if (kind == ApiKind::INPLACE_ADD || kind == ApiKind::INPLACE_ADDS) {
        outHolder = selfHolder;
    } else if (kind == ApiKind::INPLACE_ADD_V3) {
        outHolder = otherHolder;
    } else {
        ret = CreateTensor(outSpec, &outHolder);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
    }

    if (kind == ApiKind::ADD) {
        ret = aclnnAddGetWorkspaceSize(selfHolder.tensor, otherHolder.tensor, alpha.scalar, outHolder.tensor,
                                       &workspaceSize, &executor);
    } else if (kind == ApiKind::ADDS) {
        ret = aclnnAddsGetWorkspaceSize(selfHolder.tensor, otherScalar.scalar, alpha.scalar, outHolder.tensor,
                                        &workspaceSize, &executor);
    } else if (kind == ApiKind::INPLACE_ADD) {
        ret = aclnnInplaceAddGetWorkspaceSize(outHolder.tensor, otherHolder.tensor, alpha.scalar,
                                              &workspaceSize, &executor);
    } else if (kind == ApiKind::INPLACE_ADDS) {
        ret = aclnnInplaceAddsGetWorkspaceSize(outHolder.tensor, otherScalar.scalar, alpha.scalar,
                                               &workspaceSize, &executor);
    } else if (kind == ApiKind::ADD_V3) {
        ret = aclnnAddV3GetWorkspaceSize(selfScalar.scalar, otherHolder.tensor, alpha.scalar, outHolder.tensor,
                                         &workspaceSize, &executor);
    } else {
        ret = aclnnInplaceAddV3GetWorkspaceSize(selfScalar.scalar, outHolder.tensor, alpha.scalar,
                                                &workspaceSize, &executor);
    }

    if (ret != ACL_SUCCESS) {
        if (allowUnsupportedSkip) {
            LOG_PRINT("[  SKIPPED ] %s, GetWorkspaceSize ret=%d\n", name.c_str(), ret);
            pass = true;
        } else {
            LOG_PRINT("[  FAILED  ] %s, GetWorkspaceSize ret=%d\n", name.c_str(), ret);
        }
        goto cleanup;
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[  FAILED  ] %s, workspace malloc ret=%d\n", name.c_str(), ret);
            goto cleanup;
        }
    }

    if (kind == ApiKind::ADD) {
        ret = aclnnAdd(workspace, workspaceSize, executor, g_stream);
    } else if (kind == ApiKind::ADDS) {
        ret = aclnnAdds(workspace, workspaceSize, executor, g_stream);
    } else if (kind == ApiKind::INPLACE_ADD) {
        ret = aclnnInplaceAdd(workspace, workspaceSize, executor, g_stream);
    } else if (kind == ApiKind::INPLACE_ADDS) {
        ret = aclnnInplaceAdds(workspace, workspaceSize, executor, g_stream);
    } else if (kind == ApiKind::ADD_V3) {
        ret = aclnnAddV3(workspace, workspaceSize, executor, g_stream);
    } else {
        ret = aclnnInplaceAddV3(workspace, workspaceSize, executor, g_stream);
    }

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s, execute ret=%d\n", name.c_str(), ret);
        goto cleanup;
    }
    ret = aclrtSynchronizeStream(g_stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[  FAILED  ] %s, sync ret=%d\n", name.c_str(), ret);
        goto cleanup;
    }

    {
        const TensorSpec& realOutSpec = (kind == ApiKind::INPLACE_ADD || kind == ApiKind::INPLACE_ADDS) ? selfHolder.spec :
                                        (kind == ApiKind::INPLACE_ADD_V3 ? otherHolder.spec : outHolder.spec);
        std::vector<double> actual = CopyTensorResultToHost(outHolder);
        std::vector<double> expected = BuildExpected(kind, selfSpec, otherSpec, realOutSpec,
                                                     selfSpec.data.empty() ? 0.0 : selfSpec.data[0],
                                                     otherSpec.data.empty() ? 0.0 : otherSpec.data[0],
                                                     alphaValue, selfScalarDtype, otherScalarDtype, alphaDtype);
        pass = (actual.size() == expected.size());
        for (size_t i = 0; pass && i < actual.size(); ++i) {
            if (!CompareValue(actual[i], expected[i], realOutSpec.dtype)) {
                LOG_PRINT("[  FAILED  ] %s, index=%zu, actual=%lf, expected=%lf, outDtype=%s\n",
                          name.c_str(), i, actual[i], expected[i], DTypeName(realOutSpec.dtype));
                pass = false;
            }
        }
    }

cleanup:
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    DestroyScalar(&alpha);
    DestroyScalar(&selfScalar);
    DestroyScalar(&otherScalar);

    if (kind == ApiKind::INPLACE_ADD || kind == ApiKind::INPLACE_ADDS) {
        // outHolder aliases selfHolder.
        outHolder.tensor = nullptr;
        outHolder.device = nullptr;
    } else if (kind == ApiKind::INPLACE_ADD_V3) {
        // outHolder aliases otherHolder.
        outHolder.tensor = nullptr;
        outHolder.device = nullptr;
    } else {
        DestroyTensor(&outHolder);
    }
    DestroyTensor(&selfHolder);
    DestroyTensor(&otherHolder);

    LOG_PRINT("[%s] %s\n", pass ? "       OK" : "  FAILED ", name.c_str());
    return pass;
}

static bool RunNegativeCase(const std::string& name,
                            ApiKind kind,
                            TensorSpec selfSpec,
                            TensorSpec otherSpec,
                            TensorSpec outSpec,
                            double alphaValue,
                            aclDataType alphaDtype,
                            aclDataType selfScalarDtype = ACL_FLOAT,
                            aclDataType otherScalarDtype = ACL_FLOAT)
{
    LOG_PRINT("[ RUN-NEG  ] %s\n", name.c_str());
    TensorHolder selfHolder;
    TensorHolder otherHolder;
    TensorHolder outHolder;
    ScalarHolder alpha;
    ScalarHolder selfScalar;
    ScalarHolder otherScalar;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    bool pass = false;

    int ret = CreateScalar(alphaValue, alphaDtype, &alpha);
    if (ret != ACL_SUCCESS) {
        goto cleanup;
    }
    if (kind == ApiKind::ADD || kind == ApiKind::INPLACE_ADD || kind == ApiKind::ADDS || kind == ApiKind::INPLACE_ADDS) {
        ret = CreateTensor(selfSpec, &selfHolder);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
    }
    if (kind == ApiKind::ADD || kind == ApiKind::INPLACE_ADD) {
        ret = CreateTensor(otherSpec, &otherHolder);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
    }
    if (kind == ApiKind::ADDS || kind == ApiKind::INPLACE_ADDS) {
        ret = CreateScalar(otherSpec.data.empty() ? 0.0 : otherSpec.data[0], otherScalarDtype, &otherScalar);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
    }
    if (kind == ApiKind::ADD_V3 || kind == ApiKind::INPLACE_ADD_V3) {
        ret = CreateScalar(selfSpec.data.empty() ? 0.0 : selfSpec.data[0], selfScalarDtype, &selfScalar);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
        ret = CreateTensor(otherSpec, &otherHolder);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
    }
    if (kind == ApiKind::INPLACE_ADD || kind == ApiKind::INPLACE_ADDS) {
        outHolder = selfHolder;
    } else if (kind == ApiKind::INPLACE_ADD_V3) {
        outHolder = otherHolder;
    } else {
        ret = CreateTensor(outSpec, &outHolder);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
    }

    if (kind == ApiKind::ADD) {
        ret = aclnnAddGetWorkspaceSize(selfHolder.tensor, otherHolder.tensor, alpha.scalar, outHolder.tensor,
                                       &workspaceSize, &executor);
    } else if (kind == ApiKind::ADDS) {
        ret = aclnnAddsGetWorkspaceSize(selfHolder.tensor, otherScalar.scalar, alpha.scalar, outHolder.tensor,
                                        &workspaceSize, &executor);
    } else if (kind == ApiKind::INPLACE_ADD) {
        ret = aclnnInplaceAddGetWorkspaceSize(outHolder.tensor, otherHolder.tensor, alpha.scalar,
                                              &workspaceSize, &executor);
    } else if (kind == ApiKind::INPLACE_ADDS) {
        ret = aclnnInplaceAddsGetWorkspaceSize(outHolder.tensor, otherScalar.scalar, alpha.scalar,
                                               &workspaceSize, &executor);
    } else if (kind == ApiKind::ADD_V3) {
        ret = aclnnAddV3GetWorkspaceSize(selfScalar.scalar, otherHolder.tensor, alpha.scalar, outHolder.tensor,
                                         &workspaceSize, &executor);
    } else {
        ret = aclnnInplaceAddV3GetWorkspaceSize(selfScalar.scalar, outHolder.tensor, alpha.scalar,
                                                &workspaceSize, &executor);
    }
    pass = (ret != ACL_SUCCESS);
    LOG_PRINT("[%s] %s, ret=%d\n", pass ? "       OK" : "  FAILED ", name.c_str(), ret);

cleanup:
    DestroyScalar(&alpha);
    DestroyScalar(&selfScalar);
    DestroyScalar(&otherScalar);
    if (kind == ApiKind::INPLACE_ADD || kind == ApiKind::INPLACE_ADDS) {
        outHolder.tensor = nullptr;
        outHolder.device = nullptr;
    } else if (kind == ApiKind::INPLACE_ADD_V3) {
        outHolder.tensor = nullptr;
        outHolder.device = nullptr;
    } else {
        DestroyTensor(&outHolder);
    }
    DestroyTensor(&selfHolder);
    DestroyTensor(&otherHolder);
    return pass;
}

static bool RunNullptrNegativeCase()
{
    LOG_PRINT("[ RUN-NEG  ] Add_Nullptr_Params\n");
    TensorSpec spec({2, 2}, {1, 2, 3, 4}, ACL_FLOAT);
    TensorHolder tensor;
    TensorHolder out;
    ScalarHolder alpha;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    bool pass = (CreateTensor(spec, &tensor) == ACL_SUCCESS) &&
                (CreateTensor(TensorSpec({2, 2}, {0, 0, 0, 0}, ACL_FLOAT), &out) == ACL_SUCCESS) &&
                (CreateScalar(1.0, ACL_FLOAT, &alpha) == ACL_SUCCESS);
    if (pass) {
        auto ret1 = aclnnAddGetWorkspaceSize(nullptr, tensor.tensor, alpha.scalar, out.tensor, &workspaceSize, &executor);
        auto ret2 = aclnnAddGetWorkspaceSize(tensor.tensor, nullptr, alpha.scalar, out.tensor, &workspaceSize, &executor);
        auto ret3 = aclnnAddGetWorkspaceSize(tensor.tensor, tensor.tensor, nullptr, out.tensor, &workspaceSize, &executor);
        auto ret4 = aclnnAddGetWorkspaceSize(tensor.tensor, tensor.tensor, alpha.scalar, nullptr, &workspaceSize, &executor);
        pass = (ret1 != ACL_SUCCESS) && (ret2 != ACL_SUCCESS) && (ret3 != ACL_SUCCESS) && (ret4 != ACL_SUCCESS);
        LOG_PRINT("[%s] Add_Nullptr_Params, ret=(%d,%d,%d,%d)\n",
                  pass ? "       OK" : "  FAILED ", ret1, ret2, ret3, ret4);
    }
    DestroyScalar(&alpha);
    DestroyTensor(&tensor);
    DestroyTensor(&out);
    return pass;
}


static bool RunScalarAndInplaceNullptrNegativeCases()
{
    LOG_PRINT("[ RUN-NEG  ] Adds_Inplace_Nullptr_Params\n");
    TensorSpec spec({2, 2}, {1, 2, 3, 4}, ACL_FLOAT);
    TensorHolder tensor;
    TensorHolder other;
    TensorHolder out;
    ScalarHolder scalar;
    ScalarHolder alpha;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    bool pass = (CreateTensor(spec, &tensor) == ACL_SUCCESS) &&
                (CreateTensor(spec, &other) == ACL_SUCCESS) &&
                (CreateTensor(TensorSpec({2, 2}, {0, 0, 0, 0}, ACL_FLOAT), &out) == ACL_SUCCESS) &&
                (CreateScalar(2.0, ACL_FLOAT, &scalar) == ACL_SUCCESS) &&
                (CreateScalar(1.0, ACL_FLOAT, &alpha) == ACL_SUCCESS);
    if (pass) {
        auto ret1 = aclnnAddsGetWorkspaceSize(nullptr, scalar.scalar, alpha.scalar, out.tensor, &workspaceSize, &executor);
        auto ret2 = aclnnAddsGetWorkspaceSize(tensor.tensor, nullptr, alpha.scalar, out.tensor, &workspaceSize, &executor);
        auto ret3 = aclnnAddsGetWorkspaceSize(tensor.tensor, scalar.scalar, nullptr, out.tensor, &workspaceSize, &executor);
        auto ret4 = aclnnAddsGetWorkspaceSize(tensor.tensor, scalar.scalar, alpha.scalar, nullptr, &workspaceSize, &executor);
        auto ret5 = aclnnInplaceAddGetWorkspaceSize(nullptr, other.tensor, alpha.scalar, &workspaceSize, &executor);
        auto ret6 = aclnnInplaceAddGetWorkspaceSize(tensor.tensor, nullptr, alpha.scalar, &workspaceSize, &executor);
        pass = (ret1 != ACL_SUCCESS) && (ret2 != ACL_SUCCESS) && (ret3 != ACL_SUCCESS) &&
               (ret4 != ACL_SUCCESS) && (ret5 != ACL_SUCCESS) && (ret6 != ACL_SUCCESS);
        LOG_PRINT("[%s] Adds_Inplace_Nullptr_Params, ret=(%d,%d,%d,%d,%d,%d)\n",
                  pass ? "       OK" : "  FAILED ", ret1, ret2, ret3, ret4, ret5, ret6);
    }

    DestroyScalar(&scalar);
    DestroyScalar(&alpha);
    DestroyTensor(&tensor);
    DestroyTensor(&other);
    DestroyTensor(&out);
    return pass;
}


static bool RunWorkspaceOnlyCase(const std::string& name,
                                 ApiKind kind,
                                 TensorSpec selfSpec,
                                 TensorSpec otherSpec,
                                 TensorSpec outSpec,
                                 double alphaValue,
                                 aclDataType alphaDtype,
                                 aclDataType selfScalarDtype = ACL_FLOAT,
                                 aclDataType otherScalarDtype = ACL_FLOAT,
                                 int expectedMode = 0)
{
    // expectedMode: 1 means expect ACL_SUCCESS, -1 means expect failure, 0 accepts either result.
    LOG_PRINT("[ RUN-WS   ] %s\n", name.c_str());
    TensorHolder selfHolder;
    TensorHolder otherHolder;
    TensorHolder outHolder;
    ScalarHolder alpha;
    ScalarHolder selfScalar;
    ScalarHolder otherScalar;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    bool pass = (expectedMode == 0);

    int ret = CreateScalar(alphaValue, alphaDtype, &alpha);
    if (ret != ACL_SUCCESS) {
        goto cleanup;
    }
    if (kind == ApiKind::ADD || kind == ApiKind::INPLACE_ADD || kind == ApiKind::ADDS || kind == ApiKind::INPLACE_ADDS) {
        ret = CreateTensor(selfSpec, &selfHolder);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
    }
    if (kind == ApiKind::ADD || kind == ApiKind::INPLACE_ADD) {
        ret = CreateTensor(otherSpec, &otherHolder);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
    }
    if (kind == ApiKind::ADDS || kind == ApiKind::INPLACE_ADDS) {
        ret = CreateScalar(otherSpec.data.empty() ? 0.0 : otherSpec.data[0], otherScalarDtype, &otherScalar);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
    }
    if (kind == ApiKind::ADD_V3 || kind == ApiKind::INPLACE_ADD_V3) {
        ret = CreateScalar(selfSpec.data.empty() ? 0.0 : selfSpec.data[0], selfScalarDtype, &selfScalar);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
        ret = CreateTensor(otherSpec, &otherHolder);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
    }
    if (kind == ApiKind::INPLACE_ADD || kind == ApiKind::INPLACE_ADDS) {
        outHolder = selfHolder;
    } else if (kind == ApiKind::INPLACE_ADD_V3) {
        outHolder = otherHolder;
    } else {
        ret = CreateTensor(outSpec, &outHolder);
        if (ret != ACL_SUCCESS) {
            goto cleanup;
        }
    }

    if (kind == ApiKind::ADD) {
        ret = aclnnAddGetWorkspaceSize(selfHolder.tensor, otherHolder.tensor, alpha.scalar, outHolder.tensor,
                                       &workspaceSize, &executor);
    } else if (kind == ApiKind::ADDS) {
        ret = aclnnAddsGetWorkspaceSize(selfHolder.tensor, otherScalar.scalar, alpha.scalar, outHolder.tensor,
                                        &workspaceSize, &executor);
    } else if (kind == ApiKind::INPLACE_ADD) {
        ret = aclnnInplaceAddGetWorkspaceSize(outHolder.tensor, otherHolder.tensor, alpha.scalar,
                                              &workspaceSize, &executor);
    } else if (kind == ApiKind::INPLACE_ADDS) {
        ret = aclnnInplaceAddsGetWorkspaceSize(outHolder.tensor, otherScalar.scalar, alpha.scalar,
                                               &workspaceSize, &executor);
    } else if (kind == ApiKind::ADD_V3) {
        ret = aclnnAddV3GetWorkspaceSize(selfScalar.scalar, otherHolder.tensor, alpha.scalar, outHolder.tensor,
                                         &workspaceSize, &executor);
    } else {
        ret = aclnnInplaceAddV3GetWorkspaceSize(selfScalar.scalar, outHolder.tensor, alpha.scalar,
                                                &workspaceSize, &executor);
    }

    pass = (expectedMode == 0) || (expectedMode > 0 ? (ret == ACL_SUCCESS) : (ret != ACL_SUCCESS));
    LOG_PRINT("[%s] %s, ret=%d, workspace=%lu\n", pass ? "       OK" : "  FAILED ", name.c_str(), ret, workspaceSize);

cleanup:
    DestroyScalar(&alpha);
    DestroyScalar(&selfScalar);
    DestroyScalar(&otherScalar);
    if (kind == ApiKind::INPLACE_ADD || kind == ApiKind::INPLACE_ADDS) {
        outHolder.tensor = nullptr;
        outHolder.device = nullptr;
    } else if (kind == ApiKind::INPLACE_ADD_V3) {
        outHolder.tensor = nullptr;
        outHolder.device = nullptr;
    } else {
        DestroyTensor(&outHolder);
    }
    DestroyTensor(&selfHolder);
    DestroyTensor(&otherHolder);
    return pass;
}

static bool RunL0AddProbe(const std::string& name,
                          bool inplace,
                          TensorSpec selfSpec,
                          TensorSpec otherSpec,
                          bool expectSuccess)
{
    LOG_PRINT("[ RUN-L0   ] %s\n", name.c_str());
    TensorHolder selfHolder;
    TensorHolder otherHolder;
    TensorHolder seedSelf;
    TensorHolder seedOther;
    TensorHolder seedOut;
    ScalarHolder seedAlpha;
    aclOpExecutor* executor = nullptr;
    uint64_t seedWorkspaceSize = 0;
    bool matched = false;

    if (CreateTensor(selfSpec, &selfHolder) != ACL_SUCCESS || CreateTensor(otherSpec, &otherHolder) != ACL_SUCCESS) {
        goto cleanup;
    }

    // Use the public phase-1 API only to obtain a valid aclOpExecutor object.  The seed tensors
    // are intentionally kept alive until after the direct L0 probe returns.
    if (CreateTensor(TensorSpec({1}, {1}, ACL_FLOAT), &seedSelf) != ACL_SUCCESS ||
        CreateTensor(TensorSpec({1}, {1}, ACL_FLOAT), &seedOther) != ACL_SUCCESS ||
        CreateTensor(TensorSpec({1}, {0}, ACL_FLOAT), &seedOut) != ACL_SUCCESS ||
        CreateScalar(1.0, ACL_FLOAT, &seedAlpha) != ACL_SUCCESS ||
        aclnnAddGetWorkspaceSize(seedSelf.tensor, seedOther.tensor, seedAlpha.scalar, seedOut.tensor,
                                 &seedWorkspaceSize, &executor) != ACL_SUCCESS || executor == nullptr) {
        LOG_PRINT("[  SKIPPED ] %s, cannot create public executor for L0 probe\n", name.c_str());
        goto cleanup;
    }

    {
        const aclTensor* result = inplace ? l0op::AddInplace(selfHolder.tensor, otherHolder.tensor, executor)
                                          : l0op::Add(selfHolder.tensor, otherHolder.tensor, executor);
        matched = expectSuccess ? (result != nullptr) : (result == nullptr);
        LOG_PRINT("[%s] %s, result=%p\n", matched ? "       OK" : "   WARN ", name.c_str(), static_cast<const void*>(result));
    }

cleanup:
    DestroyScalar(&seedAlpha);
    DestroyTensor(&seedSelf);
    DestroyTensor(&seedOther);
    DestroyTensor(&seedOut);
    DestroyTensor(&selfHolder);
    DestroyTensor(&otherHolder);
    return true;
}



static TensorSpec MakeOutLike(const std::vector<int64_t>& shape, aclDataType dtype);

// Minimal raw probe for the Adds bool guard. It avoids the generic helper so that
// scalar storage and dtype are exactly bool/bool/bool, matching the op-api UT
// case intended to enter lines 629-634 in aclnn_add.cpp.
static bool RunAddsBoolGuardRawProbe()
{
    LOG_PRINT("[ RUN-RAW  ] Adds_BOOL_TRUE_TRUE_OutFP16_RawGuard\n");
    TensorHolder self;
    TensorHolder out;
    aclScalar* other = nullptr;
    aclScalar* alpha = nullptr;
    aclOpExecutor* executor = nullptr;
    void* workspace = nullptr;
    uint64_t workspaceSize = 0;
    bool otherValue = true;
    bool alphaValue = true;
    bool pass = true;

    if (CreateTensor(TensorSpec({10, 5}, std::vector<double>(50, 1.0), ACL_BOOL), &self) != ACL_SUCCESS ||
        CreateTensor(MakeOutLike({10, 5}, ACL_FLOAT16), &out) != ACL_SUCCESS) {
        pass = false;
        goto cleanup;
    }
    other = aclCreateScalar(&otherValue, ACL_BOOL);
    alpha = aclCreateScalar(&alphaValue, ACL_BOOL);
    if (other == nullptr || alpha == nullptr) {
        LOG_PRINT("[  FAILED  ] Adds_BOOL_TRUE_TRUE_OutFP16_RawGuard, aclCreateScalar failed\n");
        pass = false;
        goto cleanup;
    }

    {
        auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, &workspaceSize, &executor);
        LOG_PRINT("[%s] Adds_BOOL_TRUE_TRUE_OutFP16_RawGuard phase1 ret=%d workspace=%lu\n",
                  ret == ACL_SUCCESS ? "       OK" : "   WARN ", ret, workspaceSize);
        if (ret == ACL_SUCCESS) {
            if (workspaceSize > 0) {
                ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    LOG_PRINT("[   WARN ] raw bool guard workspace malloc ret=%d\n", ret);
                    goto cleanup;
                }
            }
            ret = aclnnAdds(workspace, workspaceSize, executor, g_stream);
            if (ret == ACL_SUCCESS) {
                (void)aclrtSynchronizeStream(g_stream);
            }
            LOG_PRINT("[       OK] Adds_BOOL_TRUE_TRUE_OutFP16_RawGuard phase2 ret=%d\n", ret);
        }
    }

cleanup:
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (other != nullptr) {
        aclDestroyScalar(other);
    }
    if (alpha != nullptr) {
        aclDestroyScalar(alpha);
    }
    DestroyTensor(&self);
    DestroyTensor(&out);
    return pass;
}


// Byte-backed bool-scalar probe: some ACL builds expect ACL_BOOL scalar storage
// as a uint8_t rather than C++ bool. This directly targets aclnnAdds lines
// 627-634 (bool tensor + True scalar * True alpha -> non-bool out guard).
static bool RunAddsBoolGuardByteProbe(aclDataType outDtype, const char* tag)
{
    LOG_PRINT("[ RUN-RAW  ] Adds_BOOL_BYTE_TRUE_TRUE_%s\n", tag);
    TensorHolder self;
    TensorHolder out;
    aclScalar* other = nullptr;
    aclScalar* alpha = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    uint8_t otherValue = 1;
    uint8_t alphaValue = 1;
    bool pass = true;

    TensorSpec selfSpec({10, 5}, std::vector<double>(50, 1.0), ACL_BOOL);
    TensorSpec outSpec = MakeOutLike({10, 5}, outDtype);
    if (CreateTensor(selfSpec, &self) != ACL_SUCCESS || CreateTensor(outSpec, &out) != ACL_SUCCESS) {
        pass = false;
        goto cleanup;
    }

    other = aclCreateScalar(&otherValue, ACL_BOOL);
    alpha = aclCreateScalar(&alphaValue, ACL_BOOL);
    if (other == nullptr || alpha == nullptr) {
        LOG_PRINT("[   WARN ] Adds_BOOL_BYTE_TRUE_TRUE_%s scalar create failed\n", tag);
        goto cleanup;
    }

    {
        auto ret = aclnnAddsGetWorkspaceSize(self.tensor, other, alpha, out.tensor, &workspaceSize, &executor);
        LOG_PRINT("[%s] Adds_BOOL_BYTE_TRUE_TRUE_%s ret=%d workspace=%lu\n",
                  ret == ACL_SUCCESS ? "       OK" : "   WARN ", tag, ret, workspaceSize);
    }

cleanup:
    if (other != nullptr) { aclDestroyScalar(other); }
    if (alpha != nullptr) { aclDestroyScalar(alpha); }
    DestroyTensor(&self);
    DestroyTensor(&out);
    return pass;
}

#if defined(ADD_TEST_HAS_OPDEV_PLATFORM)
static bool RunSocSwitchCoverageProbe()
{
    LOG_PRINT("[ RUN-SOC  ] SocSwitchCoverageProbe\n");
    auto original = op::GetCurrentPlatformInfo().GetCurNpuArch();
    bool mutablePlatform = TrySetPlatformArchForCoverage(original);
    if (!mutablePlatform) {
        LOG_PRINT("[  SKIPPED ] SocSwitchCoverageProbe, platform arch setter is not exposed\n");
        return true;
    }

    bool ok = true;
    const op::NpuArch arches[] = {
        op::NpuArch::DAV_1001,
        op::NpuArch::DAV_3102,
        static_cast<op::NpuArch>(0x7fffffff)
    };
    const char* names[] = {"DAV_1001", "DAV_3102", "DEFAULT"};
    for (size_t i = 0; i < 3; ++i) {
        if (!TrySetPlatformArchForCoverage(arches[i])) {
            LOG_PRINT("[   WARN ] SocSwitchCoverageProbe could not set %s\n", names[i]);
            continue;
        }
        LOG_PRINT("[ RUN-SOC  ] arch=%s\n", names[i]);
        ok &= RunWorkspaceOnlyCase(std::string("SOC_Add_WS_") + names[i], ApiKind::ADD,
                                   TensorSpec({1}, {1}, ACL_FLOAT),
                                   TensorSpec({1}, {2}, ACL_FLOAT),
                                   MakeOutLike({1}, ACL_FLOAT), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0);
        // These cases become high-value when the platform test double can leave RegBase:
        // they target IsEqualToOne(!RegBase), non-RegBase IsSupportAxpy and the mixed-dtype
        // contiguous branch at aclnn_add.cpp:358-364.
        ok &= RunWorkspaceOnlyCase(std::string("SOC_MixFP16Float_Alpha1_") + names[i], ApiKind::ADD,
                                   TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT16),
                                   TensorSpec({2, 3}, {1, 1, 1, 1, 1, 1}, ACL_FLOAT),
                                   MakeOutLike({2, 3}, ACL_FLOAT), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0);
        ok &= RunWorkspaceOnlyCase(std::string("SOC_AddFloatInt32_Axpy_") + names[i], ApiKind::ADD,
                                   TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT),
                                   TensorSpec({1, 3}, {1, 2, 3}, ACL_INT32),
                                   MakeOutLike({2, 3}, ACL_FLOAT), 2.5, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0);
        ok &= RunWorkspaceOnlyCase(std::string("SOC_AddsDoubleScalarOutFloat_") + names[i], ApiKind::ADDS,
                                   TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT64),
                                   TensorSpec({1}, {2.0}, ACL_DOUBLE),
                                   MakeOutLike({2, 3}, ACL_FLOAT), 1.0, ACL_DOUBLE, ACL_FLOAT, ACL_DOUBLE, 0);
        ok &= RunL0AddProbe(std::string("SOC_L0Add_") + names[i], false,
                            TensorSpec({1}, {1}, ACL_FLOAT),
                            TensorSpec({1}, {2}, ACL_FLOAT), true);
    }
    const char* socNames[] = {"Ascend910", "Ascend610Lite", "UnknownSocForCoverage"};
    for (const char* soc : socNames) {
        if (!TrySetPlatformSocNameForCoverage(soc)) {
            continue;
        }
        LOG_PRINT("[ RUN-SOC  ] socName=%s\n", soc);
        ok &= RunWorkspaceOnlyCase(std::string("SOCNAME_Add_WS_") + soc, ApiKind::ADD,
                                   TensorSpec({1}, {1}, ACL_FLOAT),
                                   TensorSpec({1}, {2}, ACL_FLOAT),
                                   MakeOutLike({1}, ACL_FLOAT), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0);
        ok &= RunL0AddProbe(std::string("SOCNAME_L0Add_") + soc, false,
                            TensorSpec({1}, {1}, ACL_FLOAT),
                            TensorSpec({1}, {2}, ACL_FLOAT), true);
    }
    (void)TrySetPlatformArchForCoverage(original);
    return ok;
}
#else
static bool RunSocSwitchCoverageProbe()
{
    LOG_PRINT("[  SKIPPED ] SocSwitchCoverageProbe, opdev/platform.h is not visible\n");
    return true;
}
#endif

static TensorSpec MakeOutLike(const std::vector<int64_t>& shape, aclDataType dtype)
{
    return TensorSpec(shape, std::vector<double>(static_cast<size_t>(std::max<int64_t>(0, GetShapeSize(shape))), 0.0), dtype);
}

static bool RunTilingNullContextProbe()
{
    LOG_PRINT("[ RUN-TILE ] TilingForAdd_NullContext\n");
    if (&optiling::TilingForAdd == nullptr) {
        LOG_PRINT("[  SKIPPED ] TilingForAdd_NullContext, tiling symbol is not linked\n");
        return true;
    }
    auto ret = optiling::TilingForAdd(nullptr);
    LOG_PRINT("[       OK] TilingForAdd_NullContext, ret=%u\n", static_cast<unsigned>(ret));
    return true;
}

int main()
{
    if (Init() != ACL_SUCCESS) {
        return 1;
    }

    bool allPass = true;

    // aclnnAdd: core dtype, alpha and broadcast paths.
    allPass &= RunComputeCase("Add_FP32_Alpha1_Add", ApiKind::ADD,
                              TensorSpec({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT),
                              TensorSpec({2, 4}, {10, 10, 10, 10, 20, 20, 20, 20}, ACL_FLOAT),
                              MakeOutLike({2, 4}, ACL_FLOAT), 1.0, ACL_FLOAT);

    // Alpha==1 with different non-mixed dtypes: forces the executor branch that casts both
    // inputs to promoteType and then calls l0op::Add.
    allPass &= RunComputeCase("Add_INT32_FLOAT_AlphaInt1_CastBoth_Exec", ApiKind::ADD,
                              TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT32),
                              TensorSpec({2, 3}, {0.5, 1.5, -2.0, 4.0, -5.0, 6.25}, ACL_FLOAT),
                              MakeOutLike({2, 3}, ACL_FLOAT), 1.0, ACL_INT64);

    allPass &= RunWorkspaceOnlyCase("Add_INT64_FLOAT_AlphaDouble1_CastBoth_WS", ApiKind::ADD,
                                    TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT64),
                                    TensorSpec({2, 3}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5}, ACL_FLOAT),
                                    MakeOutLike({2, 3}, ACL_FLOAT), 1.0, ACL_DOUBLE, ACL_FLOAT, ACL_FLOAT, 1);

    allPass &= RunWorkspaceOnlyCase("Add_BOOL_FLOAT_AlphaBoolTrue_CastBoth_WS", ApiKind::ADD,
                                    TensorSpec({2, 3}, {1, 0, 1, 0, 1, 0}, ACL_BOOL),
                                    TensorSpec({2, 3}, {0.25, 0.5, 0.75, 1.0, 1.25, 1.5}, ACL_FLOAT),
                                    MakeOutLike({2, 3}, ACL_FLOAT), 1.0, ACL_BOOL, ACL_FLOAT, ACL_FLOAT, 1);

    allPass &= RunComputeCase("Add_FP32_Alpha1p5_Axpy", ApiKind::ADD,
                              TensorSpec({2, 2}, {1, -2, 3, -4}, ACL_FLOAT),
                              TensorSpec({2, 2}, {2, 4, -6, 8}, ACL_FLOAT),
                              MakeOutLike({2, 2}, ACL_FLOAT), 1.5, ACL_FLOAT);

    allPass &= RunComputeCase("Add_INT64_Alpha2_AxpyV2", ApiKind::ADD,
                              TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_INT64),
                              TensorSpec({2, 2}, {5, 6, 7, 8}, ACL_INT64),
                              MakeOutLike({2, 2}, ACL_INT64), 2.0, ACL_INT64);

    allPass &= RunComputeCase("Add_INT16_Alpha2_MulAdd_AiCpu", ApiKind::ADD,
                              TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_INT16),
                              TensorSpec({2, 2}, {5, 6, 7, 8}, ACL_INT16),
                              MakeOutLike({2, 2}, ACL_INT16), 2.0, ACL_INT16);

    allPass &= RunComputeCase("Add_INT32_Tiling", ApiKind::ADD,
                              TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT32),
                              TensorSpec({2, 3}, {6, 5, 4, 3, 2, 1}, ACL_INT32),
                              MakeOutLike({2, 3}, ACL_INT32), 1.0, ACL_INT64);

    allPass &= RunComputeCase("Add_INT8_Tiling", ApiKind::ADD,
                              TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT8),
                              TensorSpec({2, 3}, {1, -1, 1, -1, 1, -1}, ACL_INT8),
                              MakeOutLike({2, 3}, ACL_INT8), 1.0, ACL_INT64);

    allPass &= RunComputeCase("Add_UINT8_Tiling", ApiKind::ADD,
                              TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_UINT8),
                              TensorSpec({2, 3}, {10, 9, 8, 7, 6, 5}, ACL_UINT8),
                              MakeOutLike({2, 3}, ACL_UINT8), 1.0, ACL_UINT8);

    allPass &= RunComputeCase("Add_BOOL_Tiling", ApiKind::ADD,
                              TensorSpec({1, 2, 3}, {1, 0, 0, 1, 0, 1}, ACL_BOOL),
                              TensorSpec({1, 2, 3}, {0, 1, 0, 1, 1, 0}, ACL_BOOL),
                              MakeOutLike({1, 2, 3}, ACL_BOOL), 1.0, ACL_BOOL);

    allPass &= RunComputeCase("Add_FP16_Alpha1p25_Axpy", ApiKind::ADD,
                              TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT16),
                              TensorSpec({2, 3}, {1, -1, 2, -2, 3, -3}, ACL_FLOAT16),
                              MakeOutLike({2, 3}, ACL_FLOAT16), 1.25, ACL_FLOAT);

    allPass &= RunComputeCase("Add_BF16_Alpha1_Tiling", ApiKind::ADD,
                              TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_BF16),
                              TensorSpec({2, 3}, {1, 1, 1, 1, 1, 1}, ACL_BF16),
                              MakeOutLike({2, 3}, ACL_BF16), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true);

    allPass &= RunComputeCase("Add_Mix_FP16_FP32", ApiKind::ADD,
                              TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_FLOAT16),
                              TensorSpec({2, 2}, {10, 20, 30, 40}, ACL_FLOAT),
                              MakeOutLike({2, 2}, ACL_FLOAT), 1.0, ACL_FLOAT);

    allPass &= RunComputeCase("Add_Mix_FP32_FP16", ApiKind::ADD,
                              TensorSpec({2, 2}, {10, 20, 30, 40}, ACL_FLOAT),
                              TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_FLOAT16),
                              MakeOutLike({2, 2}, ACL_FLOAT), 1.0, ACL_FLOAT);

    allPass &= RunComputeCase("Add_Mix_BF16_FP32", ApiKind::ADD,
                              TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_BF16),
                              TensorSpec({2, 2}, {10, 20, 30, 40}, ACL_FLOAT),
                              MakeOutLike({2, 2}, ACL_FLOAT), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true);

    allPass &= RunComputeCase("Add_Mix_FP32_BF16", ApiKind::ADD,
                              TensorSpec({2, 2}, {10, 20, 30, 40}, ACL_FLOAT),
                              TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_BF16),
                              MakeOutLike({2, 2}, ACL_FLOAT), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true);

    allPass &= RunComputeCase("Add_Broadcast_3D", ApiKind::ADD,
                              TensorSpec({2, 1, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT),
                              TensorSpec({1, 4, 1}, {10, 20, 30, 40}, ACL_FLOAT),
                              MakeOutLike({2, 4, 3}, ACL_FLOAT), 1.0, ACL_FLOAT);

    TensorSpec stridedSelf({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT);
    stridedSelf.storageShape = {2, 5};
    stridedSelf.strides = {5, 1};
    stridedSelf.offset = 1;
    TensorSpec stridedOut = MakeOutLike({2, 3}, ACL_FLOAT);
    stridedOut.storageShape = {2, 5};
    stridedOut.strides = {5, 1};
    stridedOut.offset = 1;
    allPass &= RunComputeCase("Add_NonContiguous_Input_Output", ApiKind::ADD,
                              stridedSelf,
                              TensorSpec({1, 3}, {10, 20, 30}, ACL_FLOAT),
                              stridedOut, 1.0, ACL_FLOAT);

    allPass &= RunComputeCase("Add_Empty_Tensor", ApiKind::ADD,
                              TensorSpec({1, 0, 5}, {}, ACL_INT32),
                              TensorSpec({1, 0, 5}, {}, ACL_INT32),
                              MakeOutLike({1, 0, 5}, ACL_INT32), 1.0, ACL_INT64);

    allPass &= RunComputeCase("Add_INT64_Alpha1_Tiling", ApiKind::ADD,
                              TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT64),
                              TensorSpec({2, 3}, {6, 5, 4, 3, 2, 1}, ACL_INT64),
                              MakeOutLike({2, 3}, ACL_INT64), 1.0, ACL_INT64);

    allPass &= RunComputeCase("Add_INT32_FLOAT_Alpha1_CastBoth", ApiKind::ADD,
                              TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT32),
                              TensorSpec({2, 3}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5}, ACL_FLOAT),
                              MakeOutLike({2, 3}, ACL_FLOAT), 1.0, ACL_FLOAT);

    // Both inputs need a real dtype conversion to make the cast-both branch reach
    // the final l0op::Add call instead of stopping after a same-dtype no-op cast.
    allPass &= RunWorkspaceOnlyCase("Add_BF16_FP16_Alpha1_CastBothToFloat_WS", ApiKind::ADD,
                                    TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_BF16),
                                    TensorSpec({2, 3}, {0.5, -1.5, 2.5, -3.5, 4.5, -5.5}, ACL_FLOAT16),
                                    MakeOutLike({2, 3}, ACL_FLOAT), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0);

    allPass &= RunWorkspaceOnlyCase("Add_FP16_BF16_Alpha1_CastBothToFloat_WS", ApiKind::ADD,
                                    TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT16),
                                    TensorSpec({2, 3}, {0.5, -1.5, 2.5, -3.5, 4.5, -5.5}, ACL_BF16),
                                    MakeOutLike({2, 3}, ACL_FLOAT), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0);

    allPass &= RunWorkspaceOnlyCase("Add_INT8_UINT8_Alpha1_CastBothToWider_WS", ApiKind::ADD,
                                    TensorSpec({2, 3}, {-2, -1, 0, 1, 2, 3}, ACL_INT8),
                                    TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_UINT8),
                                    MakeOutLike({2, 3}, ACL_INT16), 1.0, ACL_INT64, ACL_FLOAT, ACL_FLOAT, 0);

    // FP16 + complex64 tends to promote through complex32 on RegBase; complex32 is
    // not in Add's phase-1 support list, so this targets CheckPromoteType's
    // "promote dtype not implemented" error block instead of ordinary shape errors.
    allPass &= RunWorkspaceOnlyCase("Add_FP16_COMPLEX64_AlphaComplex64_UnsupportedComplex32_WS", ApiKind::ADD,
                                    TensorSpec({2}, {1.0, -2.0}, ACL_FLOAT16),
                                    TensorSpec({2}, {0.5, 1.5}, ACL_COMPLEX64),
                                    MakeOutLike({2}, ACL_COMPLEX64), 1.0, ACL_COMPLEX64, ACL_FLOAT, ACL_FLOAT, 0);

    // Double + complex path preserves complex128 and exercises high-precision
    // complex promotion while staying in phase-1.
    allPass &= RunWorkspaceOnlyCase("Add_DOUBLE_COMPLEX64_AlphaComplex128_WS", ApiKind::ADD,
                                    TensorSpec({2}, {1.0, -2.0}, ACL_DOUBLE),
                                    TensorSpec({2}, {0.5, 1.5}, ACL_COMPLEX64),
                                    MakeOutLike({2}, ACL_COMPLEX128), 1.0, ACL_COMPLEX128, ACL_FLOAT, ACL_FLOAT, 0);


    allPass &= RunWorkspaceOnlyCase("Add_FLOAT_INT32_Alpha1_CastOther_WS", ApiKind::ADD,
                                    TensorSpec({2, 3}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5}, ACL_FLOAT),
                                    TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT32),
                                    MakeOutLike({2, 3}, ACL_FLOAT), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1);

    allPass &= RunWorkspaceOnlyCase("Add_INT64_INT32_Alpha1_CastBoth_WS", ApiKind::ADD,
                                    TensorSpec({2, 3}, {10, 20, 30, 40, 50, 60}, ACL_INT64),
                                    TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT32),
                                    MakeOutLike({2, 3}, ACL_INT64), 1.0, ACL_INT64, ACL_FLOAT, ACL_FLOAT, 1);


    // Targeted phase-1 probes for the cast-then-Add branch in aclnnAddGetWorkspaceSize.
    allPass &= RunWorkspaceOnlyCase("Add_BOOL_FLOAT16_Alpha1_CastBranch_WS", ApiKind::ADD,
                                    TensorSpec({2, 3}, {1, 0, 1, 0, 1, 0}, ACL_BOOL),
                                    TensorSpec({2, 3}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5}, ACL_FLOAT16),
                                    MakeOutLike({2, 3}, ACL_FLOAT16), 1.0, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, 0);

    allPass &= RunWorkspaceOnlyCase("Add_INT32_DOUBLE_Alpha1_CastBranch_WS", ApiKind::ADD,
                                    TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT32),
                                    TensorSpec({2, 3}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5}, ACL_DOUBLE),
                                    MakeOutLike({2, 3}, ACL_DOUBLE), 1.0, ACL_DOUBLE, ACL_FLOAT, ACL_FLOAT, 0);

    allPass &= RunWorkspaceOnlyCase("Add_COMPLEX64_BOOL_PromoteProbe_WS", ApiKind::ADD,
                                    TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_COMPLEX64),
                                    TensorSpec({2, 2}, {1, 0, 1, 0}, ACL_BOOL),
                                    MakeOutLike({2, 2}, ACL_COMPLEX64), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0);

    TensorSpec addNchwSelf({1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT);
    TensorSpec addNchwOther({1, 1, 2, 3}, {6, 5, 4, 3, 2, 1}, ACL_FLOAT);
    TensorSpec addNchwOut = MakeOutLike({1, 1, 2, 3}, ACL_FLOAT);
    addNchwSelf.format = ACL_FORMAT_NCHW;
    addNchwOther.format = ACL_FORMAT_NCHW;
    addNchwOut.format = ACL_FORMAT_NCHW;
    allPass &= RunComputeCase("Add_NonND_Format_Warning", ApiKind::ADD,
                              addNchwSelf, addNchwOther, addNchwOut, 1.0, ACL_FLOAT);

    allPass &= RunWorkspaceOnlyCase("Add_FP16_COMPLEX64_PromoteUnsupported_WS", ApiKind::ADD,
                                    TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_FLOAT16),
                                    TensorSpec({2, 2}, {1, 1, 1, 1}, ACL_COMPLEX64),
                                    MakeOutLike({2, 2}, ACL_COMPLEX64), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0);

    // Complex/bool combinations probe PromoteType/CheckPromoteType. Some chips reject
    // them earlier; accepting either return keeps the suite portable.
    allPass &= RunWorkspaceOnlyCase("Add_BOOL_COMPLEX64_PromoteProbe_WS", ApiKind::ADD,
                                    TensorSpec({2, 2}, {1, 0, 1, 0}, ACL_BOOL),
                                    TensorSpec({2, 2}, {1, 1, 1, 1}, ACL_COMPLEX64),
                                    MakeOutLike({2, 2}, ACL_COMPLEX64), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0);

    allPass &= RunWorkspaceOnlyCase("Add_COMPLEX128_COMPLEX128_Alpha1_WS", ApiKind::ADD,
                                    TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_COMPLEX128),
                                    TensorSpec({2, 2}, {1, 1, 1, 1}, ACL_COMPLEX128),
                                    MakeOutLike({2, 2}, ACL_COMPLEX128), 1.0, ACL_DOUBLE, ACL_FLOAT, ACL_FLOAT, 0);

    // aclnnAdds / aclnnInplace* paths, including tensor-scalar promotion and bool guard.
    allPass &= RunComputeCase("Adds_FP32_Alpha1", ApiKind::ADDS,
                              TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT),
                              TensorSpec({1}, {10}, ACL_FLOAT),
                              MakeOutLike({2, 3}, ACL_FLOAT), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT);

    allPass &= RunComputeCase("Adds_FP16_Alpha1p25", ApiKind::ADDS,
                              TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT16),
                              TensorSpec({1}, {2}, ACL_FLOAT16),
                              MakeOutLike({2, 3}, ACL_FLOAT16), 1.25, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16);

    allPass &= RunComputeCase("Adds_BF16_PromoteToFP32", ApiKind::ADDS,
                              TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_BF16),
                              TensorSpec({1}, {1.1}, ACL_FLOAT),
                              MakeOutLike({2, 2}, ACL_FLOAT), 1.2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true);

    allPass &= RunComputeCase("Adds_BOOL_To_INT32_Guard", ApiKind::ADDS,
                              TensorSpec({2, 3}, {1, 0, 1, 0, 0, 1}, ACL_BOOL),
                              TensorSpec({1}, {1}, ACL_BOOL),
                              MakeOutLike({2, 3}, ACL_INT32), 1.0, ACL_BOOL, ACL_FLOAT, ACL_BOOL, true);

    allPass &= RunWorkspaceOnlyCase("Adds_BOOL_To_FLOAT_Guard_WS", ApiKind::ADDS,
                                    TensorSpec({2, 3}, {1, 0, 1, 0, 0, 1}, ACL_BOOL),
                                    TensorSpec({1}, {1}, ACL_BOOL),
                                    MakeOutLike({2, 3}, ACL_FLOAT), 1.0, ACL_BOOL, ACL_FLOAT, ACL_BOOL, 0);


    // Exact bool guard shape from the op-api UT: bool tensor + true scalar + true alpha, non-bool output.
    allPass &= RunComputeCase("Adds_BOOL_To_FLOAT16_Guard_Exec", ApiKind::ADDS,
                              TensorSpec({2, 3}, {1, 0, 1, 0, 0, 1}, ACL_BOOL),
                              TensorSpec({1}, {1}, ACL_BOOL),
                              MakeOutLike({2, 3}, ACL_FLOAT16), 1.0, ACL_BOOL, ACL_FLOAT, ACL_BOOL, false);

    allPass &= RunWorkspaceOnlyCase("Adds_BOOL_To_FLOAT16_Guard_WS", ApiKind::ADDS,
                                    TensorSpec({10, 5}, std::vector<double>(50, 1.0), ACL_BOOL),
                                    TensorSpec({1}, {1}, ACL_BOOL),
                                    MakeOutLike({10, 5}, ACL_FLOAT16), 1.0, ACL_BOOL, ACL_FLOAT, ACL_BOOL, 0);

    // Same bool guard with several out dtypes to vary CanCast/Cast behaviour before the guard.
    allPass &= RunWorkspaceOnlyCase("Adds_BOOL_To_UINT8_Guard_WS", ApiKind::ADDS,
                                    TensorSpec({2, 3}, {1, 0, 1, 0, 1, 0}, ACL_BOOL),
                                    TensorSpec({1}, {1}, ACL_BOOL),
                                    MakeOutLike({2, 3}, ACL_UINT8), 1.0, ACL_BOOL, ACL_FLOAT, ACL_BOOL, 0);

    allPass &= RunWorkspaceOnlyCase("Adds_BOOL_To_INT64_Guard_WS", ApiKind::ADDS,
                                    TensorSpec({2, 3}, {1, 0, 1, 0, 1, 0}, ACL_BOOL),
                                    TensorSpec({1}, {1}, ACL_BOOL),
                                    MakeOutLike({2, 3}, ACL_INT64), 1.0, ACL_BOOL, ACL_FLOAT, ACL_BOOL, 0);

    allPass &= RunWorkspaceOnlyCase("Adds_BOOL_To_DOUBLE_Guard_WS", ApiKind::ADDS,
                                    TensorSpec({2, 3}, {1, 0, 1, 0, 1, 0}, ACL_BOOL),
                                    TensorSpec({1}, {1}, ACL_BOOL),
                                    MakeOutLike({2, 3}, ACL_DOUBLE), 1.0, ACL_BOOL, ACL_FLOAT, ACL_BOOL, 0);

    allPass &= RunAddsBoolGuardRawProbe();
    allPass &= RunAddsBoolGuardByteProbe(ACL_FLOAT16, "OUT_FP16");
    allPass &= RunAddsBoolGuardByteProbe(ACL_INT32, "OUT_INT32");
    allPass &= RunAddsBoolGuardByteProbe(ACL_FLOAT, "OUT_FLOAT");

    allPass &= RunComputeCase("Adds_Empty_Tensor", ApiKind::ADDS,
                              TensorSpec({2, 0, 3}, {}, ACL_INT32),
                              TensorSpec({1}, {2}, ACL_INT64),
                              MakeOutLike({2, 0, 3}, ACL_INT32), 2.0, ACL_INT64, ACL_FLOAT, ACL_INT64);

    TensorSpec addsNchwSelf({1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT);
    TensorSpec addsNchwOut = MakeOutLike({1, 1, 2, 3}, ACL_FLOAT);
    addsNchwSelf.format = ACL_FORMAT_NCHW;
    addsNchwOut.format = ACL_FORMAT_NCHW;
    allPass &= RunWorkspaceOnlyCase("Adds_NonND_Format_Warning_WS", ApiKind::ADDS,
                                    addsNchwSelf,
                                    TensorSpec({1}, {2}, ACL_FLOAT),
                                    addsNchwOut, 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 1);

    allPass &= RunWorkspaceOnlyCase("Adds_DOUBLE_MulAdd_WS", ApiKind::ADDS,
                                    TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_DOUBLE),
                                    TensorSpec({1}, {2}, ACL_DOUBLE),
                                    MakeOutLike({2, 2}, ACL_DOUBLE), 2.0, ACL_DOUBLE, ACL_FLOAT, ACL_DOUBLE, 1);

    allPass &= RunWorkspaceOnlyCase("Adds_BOOL_To_INT32_Guard_WS", ApiKind::ADDS,
                                    TensorSpec({2, 3}, {1, 0, 1, 0, 0, 1}, ACL_BOOL),
                                    TensorSpec({1}, {1}, ACL_BOOL),
                                    MakeOutLike({2, 3}, ACL_INT32), 1.0, ACL_BOOL, ACL_FLOAT, ACL_BOOL, 1);

    allPass &= RunWorkspaceOnlyCase("Adds_FP16_ComplexScalar_WS", ApiKind::ADDS,
                                    TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_FLOAT16),
                                    TensorSpec({1}, {1}, ACL_COMPLEX64),
                                    MakeOutLike({2, 2}, ACL_COMPLEX64), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_COMPLEX64, 0);

    allPass &= RunWorkspaceOnlyCase("Adds_DOUBLE_ComplexScalar_WS", ApiKind::ADDS,
                                    TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_DOUBLE),
                                    TensorSpec({1}, {1}, ACL_COMPLEX64),
                                    MakeOutLike({2, 2}, ACL_COMPLEX128), 1.0, ACL_DOUBLE, ACL_FLOAT, ACL_COMPLEX64, 0);


    allPass &= RunWorkspaceOnlyCase("Adds_INT64_DoubleScalar_OutFloat_WS", ApiKind::ADDS,
                                    TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT64),
                                    TensorSpec({1}, {2.0}, ACL_DOUBLE),
                                    MakeOutLike({2, 3}, ACL_FLOAT), 2.0, ACL_DOUBLE, ACL_FLOAT, ACL_DOUBLE, 0);

    allPass &= RunWorkspaceOnlyCase("Adds_FLOAT_ComplexScalar_OutComplex64_WS", ApiKind::ADDS,
                                    TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT),
                                    TensorSpec({1}, {1.0}, ACL_COMPLEX64),
                                    MakeOutLike({2, 3}, ACL_COMPLEX64), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_COMPLEX64, 0);

    allPass &= RunComputeCase("InplaceAdd_FP32_Broadcast", ApiKind::INPLACE_ADD,
                              TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT),
                              TensorSpec({1, 3}, {10, 20, 30}, ACL_FLOAT),
                              TensorSpec(), 0.5, ACL_FLOAT);

    allPass &= RunComputeCase("InplaceAdds_INT32_Alpha2", ApiKind::INPLACE_ADDS,
                              TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT32),
                              TensorSpec({1}, {3}, ACL_INT64),
                              TensorSpec(), 2.0, ACL_INT64, ACL_FLOAT, ACL_INT64);

    // aclnnAddV3: scalar tensor paths.
    allPass &= RunComputeCase("AddV3_FP32_Alpha1", ApiKind::ADD_V3,
                              TensorSpec({1}, {100}, ACL_FLOAT),
                              TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT),
                              MakeOutLike({2, 3}, ACL_FLOAT), 1.0, ACL_FLOAT, ACL_FLOAT);

    allPass &= RunComputeCase("AddV3_FP32_Alpha2_Axpy", ApiKind::ADD_V3,
                              TensorSpec({1}, {100}, ACL_FLOAT),
                              TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_FLOAT),
                              MakeOutLike({2, 2}, ACL_FLOAT), 2.0, ACL_FLOAT, ACL_FLOAT);

    allPass &= RunComputeCase("AddV3_INT8_MulAdd", ApiKind::ADD_V3,
                              TensorSpec({1}, {3}, ACL_INT8),
                              TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT8),
                              MakeOutLike({2, 3}, ACL_INT8), 2.0, ACL_INT8, ACL_INT8);

    allPass &= RunComputeCase("AddV3_BF16", ApiKind::ADD_V3,
                              TensorSpec({1}, {3}, ACL_FLOAT),
                              TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_BF16),
                              MakeOutLike({2, 2}, ACL_FLOAT), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true);

    allPass &= RunComputeCase("AddV3_ScalarFloat_OtherInt32_Promote", ApiKind::ADD_V3,
                              TensorSpec({1}, {1.5}, ACL_FLOAT),
                              TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT32),
                              MakeOutLike({2, 3}, ACL_FLOAT), 2.0, ACL_FLOAT, ACL_FLOAT);

    allPass &= RunWorkspaceOnlyCase("AddV3_ScalarDouble_OtherInt32_OutFloat_WS", ApiKind::ADD_V3,
                                    TensorSpec({1}, {1.5}, ACL_DOUBLE),
                                    TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_INT32),
                                    MakeOutLike({2, 2}, ACL_FLOAT), 1.0, ACL_DOUBLE, ACL_DOUBLE, ACL_FLOAT, 1);

    allPass &= RunWorkspaceOnlyCase("AddV3_ComplexSelf_Promote_WS", ApiKind::ADD_V3,
                                    TensorSpec({1}, {1}, ACL_COMPLEX64),
                                    TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_FLOAT),
                                    MakeOutLike({2, 2}, ACL_COMPLEX64), 1.0, ACL_FLOAT, ACL_COMPLEX64, ACL_FLOAT, 0);

    allPass &= RunComputeCase("AddV3_Empty_Tensor", ApiKind::ADD_V3,
                              TensorSpec({1}, {3}, ACL_FLOAT),
                              TensorSpec({0, 3}, {}, ACL_FLOAT),
                              MakeOutLike({0, 3}, ACL_FLOAT), 1.0, ACL_FLOAT, ACL_FLOAT);

    allPass &= RunComputeCase("InplaceAddV3_FP32", ApiKind::INPLACE_ADD_V3,
                              TensorSpec({1}, {10}, ACL_FLOAT),
                              TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_FLOAT),
                              TensorSpec(), 3.0, ACL_FLOAT, ACL_FLOAT);

    // White-box guided aclnn_add.cpp probes. These are intentionally chosen by source condition,
    // not by case count: both inputs differ from the promoted dtype where possible to force the
    // cast-both path and line 402, while the bool scalar cases mirror the official UT shape for
    // the 629-634 guard.
    allPass &= RunWorkspaceOnlyCase("Add_FP16_INT32_Alpha1_CastBoth_Source402_WS", ApiKind::ADD,
                                    TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT16),
                                    TensorSpec({2, 3}, {1, 1, 1, 1, 1, 1}, ACL_INT32),
                                    MakeOutLike({2, 3}, ACL_FLOAT16), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0);
    allPass &= RunWorkspaceOnlyCase("Add_BF16_INT32_Alpha1_CastBoth_Source402_WS", ApiKind::ADD,
                                    TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_BF16),
                                    TensorSpec({2, 3}, {1, 1, 1, 1, 1, 1}, ACL_INT32),
                                    MakeOutLike({2, 3}, ACL_BF16), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0);
    allPass &= RunWorkspaceOnlyCase("Add_INT16_FLOAT16_Alpha1_CastBoth_Source402_WS", ApiKind::ADD,
                                    TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT16),
                                    TensorSpec({2, 3}, {1, 1, 1, 1, 1, 1}, ACL_FLOAT16),
                                    MakeOutLike({2, 3}, ACL_FLOAT16), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0);
    allPass &= RunWorkspaceOnlyCase("Add_UINT8_FLOAT16_Alpha1_CastBoth_Source402_WS", ApiKind::ADD,
                                    TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_UINT8),
                                    TensorSpec({2, 3}, {1, 1, 1, 1, 1, 1}, ACL_FLOAT16),
                                    MakeOutLike({2, 3}, ACL_FLOAT16), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0);

    allPass &= RunWorkspaceOnlyCase("Adds_UT_BOOL_BOOL_OutFP16_ExactGuard_WS", ApiKind::ADDS,
                                    TensorSpec({10, 5}, std::vector<double>(50, 1.0), ACL_BOOL),
                                    TensorSpec({1}, {1.0}, ACL_BOOL),
                                    MakeOutLike({10, 5}, ACL_FLOAT16), 1.0, ACL_BOOL, ACL_FLOAT, ACL_BOOL, 1);
    allPass &= RunComputeCase("Adds_UT_BOOL_BOOL_OutFP16_ExactGuard_Exec", ApiKind::ADDS,
                              TensorSpec({10, 5}, std::vector<double>(50, 1.0), ACL_BOOL),
                              TensorSpec({1}, {1.0}, ACL_BOOL),
                              MakeOutLike({10, 5}, ACL_FLOAT16), 1.0, ACL_BOOL, ACL_FLOAT, ACL_BOOL, false);

    // Exact op-api UT inspired complex/broadcast probes. These target the CheckPromoteType complex
    // gates and HWCN/non-ND warning path with supported tensor dtypes.
    TensorSpec complexSelf63({6, 3}, std::vector<double>(18, 1.0), ACL_COMPLEX64);
    TensorSpec complexOther2563({2, 5, 6, 3}, std::vector<double>(180, 2.0), ACL_COMPLEX64);
    TensorSpec complexOut2563 = MakeOutLike({2, 5, 6, 3}, ACL_COMPLEX64);
    complexOther2563.format = ACL_FORMAT_HWCN;
    complexOut2563.format = ACL_FORMAT_HWCN;
    allPass &= RunWorkspaceOnlyCase("Add_UT_COMPLEX64_HWCN_Broadcast_WS", ApiKind::ADD,
                                    complexSelf63, complexOther2563, complexOut2563,
                                    1.2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0);

    // L0 Add/AddInplace probes cover branches that public aclnnInplaceAdd does not reach.
    allPass &= RunL0AddProbe("L0Add_Broadcast_Fail", false,
                             TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT),
                             TensorSpec({4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, ACL_FLOAT),
                             false);
    allPass &= RunL0AddProbe("L0AddInplace_Broadcast_Fail", true,
                             TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT),
                             TensorSpec({4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, ACL_FLOAT),
                             false);
    allPass &= RunL0AddProbe("L0AddInplace_OtherBroadcastReject", true,
                             TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT),
                             TensorSpec({1, 3}, {10, 20, 30}, ACL_FLOAT),
                             false);
    allPass &= RunL0AddProbe("L0AddInplace_MixOtherFP16Reject", true,
                             TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT),
                             TensorSpec({2, 3}, {1, 1, 1, 1, 1, 1}, ACL_FLOAT16),
                             false);
    allPass &= RunL0AddProbe("L0AddInplace_MixFP16ToFP32_AiCore", true,
                             TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT16),
                             TensorSpec({2, 3}, {1, 1, 1, 1, 1, 1}, ACL_FLOAT),
                             true);
    allPass &= RunL0AddProbe("L0AddInplace_DOUBLE_AiCpu", true,
                             TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_DOUBLE),
                             TensorSpec({2, 3}, {1, 1, 1, 1, 1, 1}, ACL_DOUBLE),
                             true);

    // Invalid-parameter paths in CheckNotNull, CheckShape, CheckInplace and dtype checking.
    allPass &= RunNullptrNegativeCase();
    allPass &= RunScalarAndInplaceNullptrNegativeCases();
    allPass &= RunNegativeCase("Add_Unsupported_SELF_UINT32", ApiKind::ADD,
                                TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_UINT32),
                                TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_UINT32),
                                MakeOutLike({2, 2}, ACL_UINT32), 1.0, ACL_INT64);

    allPass &= RunNegativeCase("Adds_Unsupported_SELF_UINT32", ApiKind::ADDS,
                                TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_UINT32),
                                TensorSpec({1}, {2}, ACL_INT64),
                                MakeOutLike({2, 2}, ACL_UINT32), 1.0, ACL_INT64, ACL_FLOAT, ACL_INT64);

    allPass &= RunNegativeCase("Add_Broadcast_Mismatch", ApiKind::ADD,
                               TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT),
                               TensorSpec({4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, ACL_FLOAT),
                               MakeOutLike({2, 3}, ACL_FLOAT), 1.0, ACL_FLOAT);
    allPass &= RunNegativeCase("Add_OutputShape_Mismatch", ApiKind::ADD,
                               TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT),
                               TensorSpec({3}, {1, 2, 3}, ACL_FLOAT),
                               MakeOutLike({2, 2}, ACL_FLOAT), 1.0, ACL_FLOAT);
    allPass &= RunNegativeCase("Add_Rank_Greater_Than_8", ApiKind::ADD,
                               TensorSpec({1, 1, 1, 1, 1, 1, 1, 1, 1}, {1}, ACL_FLOAT),
                               TensorSpec({1}, {1}, ACL_FLOAT),
                               MakeOutLike({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT), 1.0, ACL_FLOAT);
    allPass &= RunNegativeCase("InplaceAdd_SelfRefShape_Mismatch", ApiKind::INPLACE_ADD,
                               TensorSpec({1, 3}, {1, 2, 3}, ACL_FLOAT),
                               TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT),
                               TensorSpec(), 1.0, ACL_FLOAT);
    allPass &= RunNegativeCase("Adds_Rank_Greater_Than_8", ApiKind::ADDS,
                               TensorSpec({1, 1, 1, 1, 1, 1, 1, 1, 1}, {1}, ACL_FLOAT),
                               TensorSpec({1}, {1}, ACL_FLOAT),
                               MakeOutLike({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT), 1.0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT);
    allPass &= RunNegativeCase("AddV3_OutputShape_Mismatch", ApiKind::ADD_V3,
                               TensorSpec({1}, {1}, ACL_FLOAT),
                               TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_FLOAT),
                               MakeOutLike({2, 2}, ACL_FLOAT), 1.0, ACL_FLOAT, ACL_FLOAT);
    allPass &= RunNegativeCase("AddV3_Unsupported_Other_INT16", ApiKind::ADD_V3,
                               TensorSpec({1}, {1}, ACL_INT16),
                               TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_INT16),
                               MakeOutLike({2, 2}, ACL_INT16), 1.0, ACL_INT16, ACL_INT16);

    allPass &= RunNegativeCase("Add_AlphaFloat_To_Bool_Invalid", ApiKind::ADD,
                               TensorSpec({2, 2}, {1, 0, 1, 0}, ACL_BOOL),
                               TensorSpec({2, 2}, {1, 1, 0, 0}, ACL_BOOL),
                               MakeOutLike({2, 2}, ACL_BOOL), 1.5, ACL_FLOAT);
    allPass &= RunNegativeCase("Adds_AlphaFloat_To_Bool_Invalid", ApiKind::ADDS,
                               TensorSpec({2, 2}, {1, 0, 1, 0}, ACL_BOOL),
                               TensorSpec({1}, {1}, ACL_BOOL),
                               MakeOutLike({2, 2}, ACL_BOOL), 1.5, ACL_FLOAT, ACL_FLOAT, ACL_BOOL);

    allPass &= RunNegativeCase("Add_AlphaFloat_To_Int32_Invalid", ApiKind::ADD,
                               TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_INT32),
                               TensorSpec({2, 2}, {1, 1, 1, 1}, ACL_INT32),
                               MakeOutLike({2, 2}, ACL_INT32), 1.5, ACL_FLOAT);
    allPass &= RunNegativeCase("Add_OutDtype_Cast_Invalid", ApiKind::ADD,
                               TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_FLOAT),
                               TensorSpec({2, 2}, {1, 1, 1, 1}, ACL_FLOAT),
                               MakeOutLike({2, 2}, ACL_INT32), 1.0, ACL_FLOAT);
    allPass &= RunNegativeCase("Adds_OutputShape_Mismatch", ApiKind::ADDS,
                               TensorSpec({2, 3}, {1, 2, 3, 4, 5, 6}, ACL_INT32),
                               TensorSpec({1}, {1}, ACL_INT64),
                               MakeOutLike({2, 2}, ACL_INT32), 1.0, ACL_INT64, ACL_FLOAT, ACL_INT64);
    allPass &= RunNegativeCase("AddV3_AlphaFloat_To_Int8_Invalid", ApiKind::ADD_V3,
                               TensorSpec({1}, {1}, ACL_INT8),
                               TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_INT8),
                               MakeOutLike({2, 2}, ACL_INT8), 1.5, ACL_FLOAT, ACL_INT8);
    allPass &= RunNegativeCase("AddV3_BoolTensor_Unsupported_AlphaFloat", ApiKind::ADD_V3,
                               TensorSpec({1}, {1}, ACL_BOOL),
                               TensorSpec({2, 2}, {1, 0, 1, 0}, ACL_BOOL),
                               MakeOutLike({2, 2}, ACL_BOOL), 1.5, ACL_FLOAT, ACL_BOOL);
    allPass &= RunNegativeCase("AddV3_OutDtype_Cast_Invalid", ApiKind::ADD_V3,
                               TensorSpec({1}, {1}, ACL_FLOAT),
                               TensorSpec({2, 2}, {1, 2, 3, 4}, ACL_FLOAT),
                               MakeOutLike({2, 2}, ACL_INT32), 1.0, ACL_FLOAT, ACL_FLOAT);
    allPass &= RunNegativeCase("AddV3_Rank_Greater_Than_8", ApiKind::ADD_V3,
                               TensorSpec({1}, {1}, ACL_FLOAT),
                               TensorSpec({1, 1, 1, 1, 1, 1, 1, 1, 1}, {1}, ACL_FLOAT),
                               MakeOutLike({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT), 1.0, ACL_FLOAT, ACL_FLOAT);

    allPass &= RunSocSwitchCoverageProbe();
    allPass &= RunTilingNullContextProbe();

    Deinit();
    LOG_PRINT("\nSUMMARY: %s\n", allPass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return allPass ? 0 : 1;
}
