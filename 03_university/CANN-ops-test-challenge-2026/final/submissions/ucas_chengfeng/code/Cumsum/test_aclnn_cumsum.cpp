#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <string>
#include "acl/acl.h"
#include "aclnnop/aclnn_cumsum.h"

#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif

// ==================== Begin: aclnn_codegen_data_utils.h ====================
#pragma once

#include <acl/acl.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace aclnn_codegen_data
{

    template <typename T>
    inline void AppendPodBytes(std::vector<uint8_t> *buffer, const T &value)
    {
        const uint8_t *src = reinterpret_cast<const uint8_t *>(&value);
        buffer->insert(buffer->end(), src, src + sizeof(T));
    }

    inline uint16_t FloatToBFloat16(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t lsb = (bits >> 16) & 1U;
        bits += 0x7FFFU + lsb;
        return static_cast<uint16_t>(bits >> 16);
    }

    inline float BFloat16ToFloat(uint16_t value)
    {
        const uint32_t bits = static_cast<uint32_t>(value) << 16;
        float result = 0.0f;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    inline uint16_t Float16ToRaw(float value)
    {
        const aclFloat16 fp16 = aclFloatToFloat16(value);
        uint16_t raw = 0;
        std::memcpy(&raw, &fp16, sizeof(raw));
        return raw;
    }

    inline float RawToFloat16(uint16_t value)
    {
        aclFloat16 fp16 = 0;
        std::memcpy(&fp16, &value, sizeof(fp16));
        return aclFloat16ToFloat(fp16);
    }

    inline uint8_t BoolToByte(bool value)
    {
        return value ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
    }

    inline bool ByteToBool(uint8_t value)
    {
        return value != 0;
    }

    inline void AppendBoolByte(std::vector<uint8_t> *buffer, bool value)
    {
        buffer->push_back(BoolToByte(value));
    }

} // namespace aclnn_codegen_data

// Compatibility wrappers for generated shard headers.
inline uint16_t FloatToBFloat16(float value)
{
    return aclnn_codegen_data::FloatToBFloat16(value);
}

inline uint16_t FloatToBfloat16(float value)
{
    return aclnn_codegen_data::FloatToBFloat16(value);
}

inline uint16_t FloatToBfloat16Raw(float value)
{
    return aclnn_codegen_data::FloatToBFloat16(value);
}

inline uint16_t FloatToBFloat16Raw(float value)
{
    return aclnn_codegen_data::FloatToBFloat16(value);
}

inline float BFloat16ToFloat(uint16_t value)
{
    return aclnn_codegen_data::BFloat16ToFloat(value);
}

inline float Bfloat16ToFloat(uint16_t value)
{
    return aclnn_codegen_data::BFloat16ToFloat(value);
}

inline float Bfloat16RawToFloat(uint16_t value)
{
    return aclnn_codegen_data::BFloat16ToFloat(value);
}

inline float Bf16ToFloatRaw(uint16_t value)
{
    return aclnn_codegen_data::BFloat16ToFloat(value);
}

inline uint16_t FloatToFloat16(float value)
{
    return aclnn_codegen_data::Float16ToRaw(value);
}

inline float Float16ToFloat(uint16_t value)
{
    return aclnn_codegen_data::RawToFloat16(value);
}

inline uint8_t BoolToByte(bool value)
{
    return aclnn_codegen_data::BoolToByte(value);
}

inline bool ByteToBool(uint8_t value)
{
    return aclnn_codegen_data::ByteToBool(value);
}
// ==================== End: aclnn_codegen_data_utils.h ====================

// ---------- Shared inline helpers available to shard headers ----------

static aclnnStatus CheckAclnnStatus(aclnnStatus status, const char *callName)
{
    if (status != ACLNN_SUCCESS)
    {
        std::cerr << "[ERROR] " << callName << " failed, status = " << static_cast<int32_t>(status) << std::endl;
    }
    return status;
}

static aclTensor *CreateAclTensorND(
    const std::vector<int64_t> &shape,
    aclDataType dataType,
    void *devicePtr)
{
    int64_t numel = 1;
    for (auto d : shape)
    {
        numel *= d;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return aclCreateTensor(
        shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        devicePtr);
}

static aclScalar *CreateAclScalarInt64(int64_t value)
{
    aclDataType dtype = ACL_INT64;
    return aclCreateScalar(&value, dtype);
}

static aclScalar *CreateAclScalarBool(bool value)
{
    uint8_t v = value ? 1 : 0;
    aclDataType dtype = ACL_BOOL;
    return aclCreateScalar(&v, dtype);
}

static aclIntArray *CreateAclIntArray(const std::vector<int64_t> &values)
{
    return aclCreateIntArray(values.data(), values.size());
}

// ---------- Shard headers (wrapped in namespaces) ----------

namespace cumsum_v1
{

    static size_t GetElementCount(const std::vector<int64_t> &shape)
    {
        if (shape.empty())
            return 1; // 0-dim scalar
        size_t count = 1;
        for (auto d : shape)
            count *= d;
        return count;
    }

    static size_t GetDataTypeSize(aclDataType dt)
    {
        switch (dt)
        {
        case ACL_FLOAT:
            return sizeof(float);
        case ACL_DOUBLE:
            return sizeof(double);
        case ACL_FLOAT16:
            return sizeof(uint16_t);
        case ACL_BF16:
            return sizeof(uint16_t);
        case ACL_INT8:
            return sizeof(int8_t);
        case ACL_INT16:
            return sizeof(int16_t);
        case ACL_INT32:
            return sizeof(int32_t);
        case ACL_INT64:
            return sizeof(int64_t);
        case ACL_UINT8:
            return sizeof(uint8_t);
        case ACL_BOOL:
            return sizeof(uint8_t);
        default:
            return 0;
        }
    }

    static std::vector<uint8_t> ConvertFloatToRawBytes(const std::vector<float> &values, aclDataType dt)
    {
        size_t elemSize = GetDataTypeSize(dt);
        std::vector<uint8_t> bytes(values.size() * elemSize);
        for (size_t i = 0; i < values.size(); ++i)
        {
            float v = values[i];
            uint8_t *dst = bytes.data() + i * elemSize;
            switch (dt)
            {
            case ACL_FLOAT:
            {
                float fv = static_cast<float>(v);
                memcpy(dst, &fv, sizeof(float));
                break;
            }
            case ACL_DOUBLE:
            {
                double dv = static_cast<double>(v);
                memcpy(dst, &dv, sizeof(double));
                break;
            }
            case ACL_FLOAT16:
            {
                uint16_t raw = FloatToFloat16(v);
                memcpy(dst, &raw, sizeof(uint16_t));
                break;
            }
            case ACL_BF16:
            {
                uint16_t raw = FloatToBFloat16(v);
                memcpy(dst, &raw, sizeof(uint16_t));
                break;
            }
            case ACL_INT8:
            {
                int8_t iv = static_cast<int8_t>(v);
                memcpy(dst, &iv, sizeof(int8_t));
                break;
            }
            case ACL_INT16:
            {
                int16_t iv = static_cast<int16_t>(v);
                memcpy(dst, &iv, sizeof(int16_t));
                break;
            }
            case ACL_INT32:
            {
                int32_t iv = static_cast<int32_t>(v);
                memcpy(dst, &iv, sizeof(int32_t));
                break;
            }
            case ACL_INT64:
            {
                int64_t iv = static_cast<int64_t>(v);
                memcpy(dst, &iv, sizeof(int64_t));
                break;
            }
            case ACL_UINT8:
            {
                uint8_t uv = static_cast<uint8_t>(v);
                memcpy(dst, &uv, sizeof(uint8_t));
                break;
            }
            case ACL_BOOL:
            {
                uint8_t bv = BoolToByte(v != 0.0f);
                memcpy(dst, &bv, sizeof(uint8_t));
                break;
            }
            default:
                break;
            }
        }
        return bytes;
    }

    static std::vector<float> ConvertRawBytesToFloat(const std::vector<uint8_t> &bytes, aclDataType dt)
    {
        size_t elemSize = GetDataTypeSize(dt);
        size_t count = bytes.size() / elemSize;
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i)
        {
            const uint8_t *src = bytes.data() + i * elemSize;
            switch (dt)
            {
            case ACL_FLOAT:
            {
                float fv;
                memcpy(&fv, src, sizeof(float));
                values[i] = fv;
                break;
            }
            case ACL_DOUBLE:
            {
                double dv;
                memcpy(&dv, src, sizeof(double));
                values[i] = static_cast<float>(dv);
                break;
            }
            case ACL_FLOAT16:
            {
                uint16_t raw;
                memcpy(&raw, src, sizeof(uint16_t));
                values[i] = Float16ToFloat(raw);
                break;
            }
            case ACL_BF16:
            {
                uint16_t raw;
                memcpy(&raw, src, sizeof(uint16_t));
                values[i] = BFloat16ToFloat(raw);
                break;
            }
            case ACL_INT8:
            {
                int8_t iv;
                memcpy(&iv, src, sizeof(int8_t));
                values[i] = static_cast<float>(iv);
                break;
            }
            case ACL_INT16:
            {
                int16_t iv;
                memcpy(&iv, src, sizeof(int16_t));
                values[i] = static_cast<float>(iv);
                break;
            }
            case ACL_INT32:
            {
                int32_t iv;
                memcpy(&iv, src, sizeof(int32_t));
                values[i] = static_cast<float>(iv);
                break;
            }
            case ACL_INT64:
            {
                int64_t iv;
                memcpy(&iv, src, sizeof(int64_t));
                values[i] = static_cast<float>(iv);
                break;
            }
            case ACL_UINT8:
            {
                uint8_t uv;
                memcpy(&uv, src, sizeof(uint8_t));
                values[i] = static_cast<float>(uv);
                break;
            }
            case ACL_BOOL:
            {
                uint8_t bv;
                memcpy(&bv, src, sizeof(uint8_t));
                values[i] = ByteToBool(bv) ? 1.0f : 0.0f;
                break;
            }
            default:
                values[i] = 0.0f;
                break;
            }
        }
        return values;
    }

    static aclTensor *CreateContiguousTensor(
        const std::vector<int64_t> &shape,
        aclDataType dt,
        void *devPtr)
    {
        size_t count = GetElementCount(shape);
        std::vector<int64_t> strides(shape.size());
        if (!shape.empty())
        {
            strides[shape.size() - 1] = 1;
            for (size_t i = shape.size() - 1; i > 0; --i)
            {
                strides[i - 1] = strides[i] * shape[i];
            }
        }
        return aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), devPtr);
    }

    static int RunPositiveCase(
        aclrtStream stream,
        const std::string &caseId,
        const std::vector<int64_t> &selfShape,
        aclDataType selfDtype,
        int64_t dim,
        aclDataType dtypeParam,
        const std::vector<int64_t> &outShape,
        aclDataType outDtype,
        const std::vector<float> &inputValues,
        const std::vector<float> &expectedValues,
        float absTol,
        bool exactMatch)
    {
        size_t selfCount = GetElementCount(selfShape);
        size_t outCount = GetElementCount(outShape);
        size_t selfByteSize = selfCount * GetDataTypeSize(selfDtype);
        size_t outByteSize = outCount * GetDataTypeSize(outDtype);

        std::vector<uint8_t> hostSelf = ConvertFloatToRawBytes(inputValues, selfDtype);
        std::vector<uint8_t> hostOut(outByteSize, 0);
        std::vector<uint8_t> hostExpected = ConvertFloatToRawBytes(expectedValues, outDtype);

        void *devSelf = nullptr;
        void *devOut = nullptr;

        if (selfByteSize > 0)
        {
            aclError aclRet = aclrtMalloc(&devSelf, selfByteSize, ACL_MEM_MALLOC_NORMAL_ONLY);
            if (aclRet != ACL_SUCCESS)
            {
                printf("[%s] aclrtMalloc self failed, ret=%d\n", caseId.c_str(), aclRet);
                return 1;
            }
            aclRet = aclrtMemcpy(devSelf, selfByteSize, hostSelf.data(), selfByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
            if (aclRet != ACL_SUCCESS)
            {
                printf("[%s] aclrtMemcpy self failed, ret=%d\n", caseId.c_str(), aclRet);
                aclrtFree(devSelf);
                return 1;
            }
        }

        if (outByteSize > 0)
        {
            aclError aclRet = aclrtMalloc(&devOut, outByteSize, ACL_MEM_MALLOC_NORMAL_ONLY);
            if (aclRet != ACL_SUCCESS)
            {
                printf("[%s] aclrtMalloc out failed, ret=%d\n", caseId.c_str(), aclRet);
                if (devSelf)
                    aclrtFree(devSelf);
                return 1;
            }
        }

        aclTensor *selfTensor = CreateContiguousTensor(selfShape, selfDtype, devSelf);
        aclTensor *outTensor = CreateContiguousTensor(outShape, outDtype, devOut);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        aclnnStatus ret = aclnnCumsumGetWorkspaceSize(selfTensor, dim, dtypeParam, outTensor, &workspaceSize, &executor);
        if (ret != ACLNN_SUCCESS)
        {
            printf("[%s] aclnnCumsumGetWorkspaceSize failed, ret=%d\n", caseId.c_str(), ret);
            aclDestroyTensor(selfTensor);
            aclDestroyTensor(outTensor);
            if (devSelf)
                aclrtFree(devSelf);
            if (devOut)
                aclrtFree(devOut);
            return 1;
        }

        void *workspace = nullptr;
        if (workspaceSize > 0)
        {
            aclError aclRet = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
            if (aclRet != ACL_SUCCESS)
            {
                printf("[%s] aclrtMalloc workspace failed, ret=%d\n", caseId.c_str(), aclRet);
                aclDestroyTensor(selfTensor);
                aclDestroyTensor(outTensor);
                if (devSelf)
                    aclrtFree(devSelf);
                if (devOut)
                    aclrtFree(devOut);
                return 1;
            }
        }

        ret = aclnnCumsum(workspace, workspaceSize, executor, stream);
        if (ret != ACLNN_SUCCESS)
        {
            printf("[%s] aclnnCumsum failed, ret=%d\n", caseId.c_str(), ret);
            if (workspace)
                aclrtFree(workspace);
            aclDestroyTensor(selfTensor);
            aclDestroyTensor(outTensor);
            if (devSelf)
                aclrtFree(devSelf);
            if (devOut)
                aclrtFree(devOut);
            return 1;
        }

        aclError aclRet = aclrtSynchronizeStream(stream);
        if (aclRet != ACL_SUCCESS)
        {
            printf("[%s] aclrtSynchronizeStream failed, ret=%d\n", caseId.c_str(), aclRet);
            if (workspace)
                aclrtFree(workspace);
            aclDestroyTensor(selfTensor);
            aclDestroyTensor(outTensor);
            if (devSelf)
                aclrtFree(devSelf);
            if (devOut)
                aclrtFree(devOut);
            return 1;
        }

        if (outByteSize > 0)
        {
            aclRet = aclrtMemcpy(hostOut.data(), outByteSize, devOut, outByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
            if (aclRet != ACL_SUCCESS)
            {
                printf("[%s] aclrtMemcpy out failed, ret=%d\n", caseId.c_str(), aclRet);
                if (workspace)
                    aclrtFree(workspace);
                aclDestroyTensor(selfTensor);
                aclDestroyTensor(outTensor);
                if (devSelf)
                    aclrtFree(devSelf);
                if (devOut)
                    aclrtFree(devOut);
                return 1;
            }
        }

        bool passed = true;
        double maxError = 0.0;
        size_t maxErrorPos = 0;
        if (exactMatch)
        {
            if (hostOut != hostExpected)
            {
                passed = false;
            }
        }
        else
        {
            std::vector<float> actualFloats = ConvertRawBytesToFloat(hostOut, outDtype);
            std::vector<float> expectedFloats = ConvertRawBytesToFloat(hostExpected, outDtype);
            if (actualFloats.size() != expectedFloats.size())
            {
                passed = false;
            }
            else
            {
                for (size_t i = 0; i < actualFloats.size(); ++i)
                {
                    double err = std::fabs(actualFloats[i] - expectedFloats[i]);
                    if (err > maxError)
                    {
                        maxError = err;
                        maxErrorPos = i;
                    }
                    if (err > absTol)
                    {
                        passed = false;
                    }
                }
            }
        }

        // 输出详细信息
        if (passed)
        {
            printf("[PASS] %s\n", caseId.c_str());
            if (!exactMatch && maxError > 0)
            {
                printf("  Max error: %.9f (at position %zu, tolerance=%.6f)\n",
                       maxError, maxErrorPos, absTol);
            }
            // 可选：打印前几个预期值与实际值
            std::vector<float> actualFloats = ConvertRawBytesToFloat(hostOut, outDtype);
            std::vector<float> expectedFloats = ConvertRawBytesToFloat(hostExpected, outDtype);
            printf("  Sample (first 5):\n");
            for (size_t i = 0; i < std::min((size_t)5, actualFloats.size()); ++i)
            {
                printf("    idx %zu: expected=%.9f, actual=%.9f\n",
                       i, expectedFloats[i], actualFloats[i]);
            }
        }
        else
        {
            printf("[FAIL] %s\n", caseId.c_str());
            if (!exactMatch)
            {
                std::vector<float> actualFloats = ConvertRawBytesToFloat(hostOut, outDtype);
                std::vector<float> expectedFloats = ConvertRawBytesToFloat(hostExpected, outDtype);
                printf("  Max error: %.9f at position %zu (tolerance=%.6f)\n",
                       maxError, maxErrorPos, absTol);
                printf("  Expected vs Actual (first 10 mismatches if any):\n");
                int mismatchPrinted = 0;
                for (size_t i = 0; i < actualFloats.size() && mismatchPrinted < 10; ++i)
                {
                    double err = std::fabs(actualFloats[i] - expectedFloats[i]);
                    if (err > absTol)
                    {
                        printf("    idx %zu: expected=%.9f, actual=%.9f (err=%.9f)\n",
                               i, expectedFloats[i], actualFloats[i], err);
                        mismatchPrinted++;
                    }
                }
            }
            else
            {
                // exact match 失败时，直接逐元素对比
                printf("  Exact match failed:\n");
                // 打印差异...
            }
        }
        if (workspace)
            aclrtFree(workspace);
        aclDestroyTensor(selfTensor);
        aclDestroyTensor(outTensor);
        if (devSelf)
            aclrtFree(devSelf);
        if (devOut)
            aclrtFree(devOut);

        return passed ? 0 : 1;
    }

    static int RunErrorCaseNullSelf(aclrtStream stream)
    {
        std::vector<int64_t> outShape = {2, 3};
        size_t outByteSize = 6 * sizeof(float);
        void *devOut = nullptr;
        aclError aclRet = aclrtMalloc(&devOut, outByteSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (aclRet != ACL_SUCCESS)
        {
            printf("[err_null_self] aclrtMalloc failed, ret=%d\n", aclRet);
            return 1;
        }
        aclTensor *outTensor = CreateContiguousTensor(outShape, ACL_FLOAT, devOut);

        uint64_t ws = 0;
        aclOpExecutor *ex = nullptr;
        aclnnStatus ret = aclnnCumsumGetWorkspaceSize(nullptr, 0, ACL_FLOAT, outTensor, &ws, &ex);

        int fail = 0;
        if (ret == ACLNN_SUCCESS)
        {
            printf("[FAIL] err_null_self: Expected non-success but got ACLNN_SUCCESS\n");
            fail = 1;
        }
        else
        {
            printf("[PASS] err_null_self: Correctly rejected null self tensor (status=%d)\n", ret);
        }
        aclDestroyTensor(outTensor);
        aclrtFree(devOut);
        return fail;
    }

    static int RunErrorCaseNullOut(aclrtStream stream)
    {
        std::vector<int64_t> selfShape = {2, 3};
        size_t selfByteSize = 6 * sizeof(float);
        void *devSelf = nullptr;
        aclError aclRet = aclrtMalloc(&devSelf, selfByteSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (aclRet != ACL_SUCCESS)
        {
            printf("[err_null_out] aclrtMalloc failed, ret=%d\n", aclRet);
            return 1;
        }
        aclTensor *selfTensor = CreateContiguousTensor(selfShape, ACL_FLOAT, devSelf);

        uint64_t ws = 0;
        aclOpExecutor *ex = nullptr;
        aclnnStatus ret = aclnnCumsumGetWorkspaceSize(selfTensor, 0, ACL_FLOAT, nullptr, &ws, &ex);

        int fail = 0;
        if (ret == ACLNN_SUCCESS)
        {
            printf("[err_null_out] Expected non-success, got ACLNN_SUCCESS\n");
            fail = 1;
        }
        else
        {
            printf("[err_null_out] Passed, got %d\n", ret);
        }

        aclDestroyTensor(selfTensor);
        aclrtFree(devSelf);
        return fail;
    }

    static int RunErrorCaseDimOutOfRange(aclrtStream stream)
    {
        std::vector<int64_t> shape = {2, 3};
        size_t byteSize = 6 * sizeof(float);
        void *devSelf = nullptr;
        void *devOut = nullptr;
        aclError aclRet1 = aclrtMalloc(&devSelf, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        aclError aclRet2 = aclrtMalloc(&devOut, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (aclRet1 != ACL_SUCCESS || aclRet2 != ACL_SUCCESS)
        {
            printf("[err_dim_out_of_range] aclrtMalloc failed\n");
            if (devSelf)
                aclrtFree(devSelf);
            if (devOut)
                aclrtFree(devOut);
            return 1;
        }

        aclTensor *selfTensor = CreateContiguousTensor(shape, ACL_FLOAT, devSelf);
        aclTensor *outTensor = CreateContiguousTensor(shape, ACL_FLOAT, devOut);

        uint64_t ws = 0;
        aclOpExecutor *ex = nullptr;
        aclnnStatus ret = aclnnCumsumGetWorkspaceSize(selfTensor, 5, ACL_FLOAT, outTensor, &ws, &ex);

        int fail = 0;
        if (ret == ACLNN_SUCCESS)
        {
            printf("[err_dim_out_of_range] Expected non-success, got ACLNN_SUCCESS\n");
            fail = 1;
        }
        else
        {
            printf("[err_dim_out_of_range] Passed, got %d\n", ret);
        }

        aclDestroyTensor(selfTensor);
        aclDestroyTensor(outTensor);
        aclrtFree(devSelf);
        aclrtFree(devOut);
        return fail;
    }

    static int RunErrorCaseShapeMismatch(aclrtStream stream)
    {
        std::vector<int64_t> selfShape = {2, 3};
        std::vector<int64_t> outShape = {3, 2};
        size_t selfByteSize = 6 * sizeof(float);
        size_t outByteSize = 6 * sizeof(float);
        void *devSelf = nullptr;
        void *devOut = nullptr;
        aclError aclRet1 = aclrtMalloc(&devSelf, selfByteSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        aclError aclRet2 = aclrtMalloc(&devOut, outByteSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (aclRet1 != ACL_SUCCESS || aclRet2 != ACL_SUCCESS)
        {
            printf("[err_shape_mismatch] aclrtMalloc failed\n");
            if (devSelf)
                aclrtFree(devSelf);
            if (devOut)
                aclrtFree(devOut);
            return 1;
        }

        aclTensor *selfTensor = CreateContiguousTensor(selfShape, ACL_FLOAT, devSelf);
        aclTensor *outTensor = CreateContiguousTensor(outShape, ACL_FLOAT, devOut);

        uint64_t ws = 0;
        aclOpExecutor *ex = nullptr;
        aclnnStatus ret = aclnnCumsumGetWorkspaceSize(selfTensor, 1, ACL_FLOAT, outTensor, &ws, &ex);

        int fail = 0;
        if (ret == ACLNN_SUCCESS)
        {
            printf("[err_shape_mismatch] Expected non-success, got ACLNN_SUCCESS\n");
            fail = 1;
        }
        else
        {
            printf("[err_shape_mismatch] Passed, got %d\n", ret);
        }

        aclDestroyTensor(selfTensor);
        aclDestroyTensor(outTensor);
        aclrtFree(devSelf);
        aclrtFree(devOut);
        return fail;
    }

    static int RunErrorCaseDtypeParamMismatch(aclrtStream stream)
    {
        std::vector<int64_t> shape = {4};
        size_t byteSize = 4 * sizeof(float);
        void *devSelf = nullptr;
        void *devOut = nullptr;
        aclError aclRet1 = aclrtMalloc(&devSelf, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        aclError aclRet2 = aclrtMalloc(&devOut, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (aclRet1 != ACL_SUCCESS || aclRet2 != ACL_SUCCESS)
        {
            printf("[err_dtype_param_mismatch] aclrtMalloc failed\n");
            if (devSelf)
                aclrtFree(devSelf);
            if (devOut)
                aclrtFree(devOut);
            return 1;
        }

        aclTensor *selfTensor = CreateContiguousTensor(shape, ACL_FLOAT, devSelf);
        aclTensor *outTensor = CreateContiguousTensor(shape, ACL_FLOAT, devOut);

        uint64_t ws = 0;
        aclOpExecutor *ex = nullptr;
        aclnnStatus ret = aclnnCumsumGetWorkspaceSize(selfTensor, 0, ACL_INT32, outTensor, &ws, &ex);

        int fail = 0;
        if (ret == ACLNN_SUCCESS)
        {
            printf("[err_dtype_param_mismatch] Expected non-success, got ACLNN_SUCCESS\n");
            fail = 1;
        }
        else
        {
            printf("[err_dtype_param_mismatch] Passed, got %d\n", ret);
        }

        aclDestroyTensor(selfTensor);
        aclDestroyTensor(outTensor);
        aclrtFree(devSelf);
        aclrtFree(devOut);
        return fail;
    }

    static int RunErrorCaseUnsupportedOutDtypeBool(aclrtStream stream)
    {
        std::vector<int64_t> shape = {4};
        size_t byteSize = 4 * sizeof(uint8_t);
        void *devSelf = nullptr;
        void *devOut = nullptr;
        aclError aclRet1 = aclrtMalloc(&devSelf, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        aclError aclRet2 = aclrtMalloc(&devOut, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (aclRet1 != ACL_SUCCESS || aclRet2 != ACL_SUCCESS)
        {
            printf("[err_unsupported_out_dtype_bool] aclrtMalloc failed\n");
            if (devSelf)
                aclrtFree(devSelf);
            if (devOut)
                aclrtFree(devOut);
            return 1;
        }

        aclTensor *selfTensor = CreateContiguousTensor(shape, ACL_BOOL, devSelf);
        aclTensor *outTensor = CreateContiguousTensor(shape, ACL_BOOL, devOut);

        uint64_t ws = 0;
        aclOpExecutor *ex = nullptr;
        aclnnStatus ret = aclnnCumsumGetWorkspaceSize(selfTensor, 0, ACL_BOOL, outTensor, &ws, &ex);

        int fail = 0;
        if (ret == ACLNN_SUCCESS)
        {
            printf("[err_unsupported_out_dtype_bool] Expected non-success, got ACLNN_SUCCESS\n");
            fail = 1;
        }
        else
        {
            printf("[err_unsupported_out_dtype_bool] Passed, got %d\n", ret);
        }

        aclDestroyTensor(selfTensor);
        aclDestroyTensor(outTensor);
        aclrtFree(devSelf);
        aclrtFree(devOut);
        return fail;
    }

    int RunShard_cumsum_v1(aclrtStream stream)
    {
        int failCount = 0;

        // Normal scenarios
        failCount += RunPositiveCase(stream, "v1_1d_dim0_float", {5}, ACL_FLOAT, 0, ACL_FLOAT, {5}, ACL_FLOAT, {1.0, 2.0, 3.0, 4.0, 5.0}, {1.0, 3.0, 6.0, 10.0, 15.0}, 1e-5, false);
        failCount += RunPositiveCase(stream, "v1_2d_dim1_int32", {2, 3}, ACL_INT32, 1, ACL_INT32, {2, 3}, ACL_INT32, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {1.0, 3.0, 6.0, 4.0, 9.0, 15.0}, 0, true);
        failCount += RunPositiveCase(stream, "v1_2d_dim0_int64", {2, 3}, ACL_INT64, 0, ACL_INT64, {2, 3}, ACL_INT64, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {1.0, 2.0, 3.0, 5.0, 7.0, 9.0}, 0, true);
        failCount += RunPositiveCase(stream, "v1_2d_neg_dim_float", {2, 3}, ACL_FLOAT, -1, ACL_FLOAT, {2, 3}, ACL_FLOAT, {-3.0, -1.0, 0.0, 2.0, 5.0, 7.0}, {-3.0, -4.0, -4.0, 2.0, 7.0, 14.0}, 1e-5, false);
        failCount += RunPositiveCase(stream, "v1_0dim_scalar", {}, ACL_FLOAT, 0, ACL_FLOAT, {}, ACL_FLOAT, {5.0}, {5.0}, 1e-5, false);
        failCount += RunPositiveCase(stream, "v1_empty_tensor", {2, 0}, ACL_FLOAT, 1, ACL_FLOAT, {2, 0}, ACL_FLOAT, {}, {}, 0, true);

        // Dtype scenarios
        failCount += RunPositiveCase(stream, "dtype_float16_lowprec", {4}, ACL_FLOAT16, 0, ACL_FLOAT16, {4}, ACL_FLOAT16, {1.0, 2.0, 3.0, 4.0}, {1.0, 3.0, 6.0, 10.0}, 0.1, false);
        failCount += RunPositiveCase(stream, "dtype_bf16_910b", {4}, ACL_BF16, 0, ACL_BF16, {4}, ACL_BF16, {1.0, 2.0, 3.0, 4.0}, {1.0, 3.0, 6.0, 10.0}, 0.1, false);
        failCount += RunPositiveCase(stream, "dtype_mixed_int2float_cast", {4}, ACL_INT8, 0, ACL_FLOAT, {4}, ACL_FLOAT, {1.0, 2.0, 3.0, 4.0}, {1.0, 3.0, 6.0, 10.0}, 1e-5, false);
        failCount += RunPositiveCase(stream, "dtype_double_prec", {4}, ACL_DOUBLE, 0, ACL_DOUBLE, {4}, ACL_DOUBLE, {1.0, 2.0, 3.0, 4.0}, {1.0, 3.0, 6.0, 10.0}, 1e-9, false);

        // Error scenarios
        failCount += RunErrorCaseNullSelf(stream);
        failCount += RunErrorCaseNullOut(stream);
        failCount += RunErrorCaseDimOutOfRange(stream);
        failCount += RunErrorCaseShapeMismatch(stream);
        failCount += RunErrorCaseDtypeParamMismatch(stream);
        failCount += RunErrorCaseUnsupportedOutDtypeBool(stream);

        return failCount;
    }

} // namespace cumsum_v1

namespace cumsum_v2
{

    static std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &shape)
    {
        if (shape.empty())
            return {};
        std::vector<int64_t> strides(shape.size());
        strides.back() = 1;
        for (size_t i = shape.size() - 1; i > 0; --i)
        {
            strides[i - 1] = strides[i] * shape[i];
        }
        return strides;
    }

    static size_t ComputeElementCount(const std::vector<int64_t> &shape)
    {
        if (shape.empty())
            return 1;
        size_t count = 1;
        for (auto d : shape)
            count *= static_cast<size_t>(d);
        return count;
    }

    static size_t GetDtypeElementSize(aclDataType dtype)
    {
        switch (dtype)
        {
        case ACL_FLOAT:
            return 4;
        case ACL_FLOAT16:
            return 2;
        case ACL_BF16:
            return 2;
        case ACL_INT32:
            return 4;
        case ACL_INT64:
            return 8;
        case ACL_DOUBLE:
            return 8;
        case ACL_UINT8:
            return 1;
        case ACL_INT8:
            return 1;
        case ACL_INT16:
            return 2;
        case ACL_COMPLEX64:
            return 8;
        case ACL_COMPLEX128:
            return 16;
        default:
            return 0;
        }
    }

    static int RunCumsumV2Op(aclTensor *self, int64_t dim, bool exclusive, bool reverse,
                             aclTensor *out, aclrtStream stream)
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus ret = aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out,
                                                        &workspaceSize, &executor);
        if (ret != ACLNN_SUCCESS)
        {
            return static_cast<int>(ret);
        }
        void *workspace = nullptr;
        if (workspaceSize > 0)
        {
            aclError aclRet = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
            if (aclRet != ACL_SUCCESS)
            {
                return -1;
            }
        }
        ret = aclnnCumsumV2(workspace, workspaceSize, executor, stream);
        if (workspace != nullptr)
        {
            aclrtFree(workspace);
        }
        if (ret != ACLNN_SUCCESS)
        {
            return static_cast<int>(ret);
        }
        aclrtSynchronizeStream(stream);
        return 0;
    }

    static aclTensor *CreateContiguousTensor(const std::vector<int64_t> &shape,
                                             aclDataType dtype, void *devPtr)
    {
        std::vector<int64_t> strides = ComputeStrides(shape);
        return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                               ACL_FORMAT_ND, shape.data(), shape.size(), devPtr);
    }

    // ===== ACL_FLOAT test =====
    static int RunCumsumV2FloatTest(const std::vector<int64_t> &shape,
                                    const std::vector<float> &inputVals,
                                    const std::vector<float> &expectedVals,
                                    int64_t dim, bool exclusive, bool reverse,
                                    float tolerance, aclrtStream stream)
    {
        size_t elemCount = ComputeElementCount(shape);
        if (elemCount == 0)
            return 0;
        size_t byteSize = elemCount * sizeof(float);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, inputVals.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_FLOAT, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_FLOAT, outDev);

        int opRet = RunCumsumV2Op(self, dim, exclusive, reverse, out, stream);

        std::vector<float> result(elemCount);
        aclrtMemcpy(result.data(), byteSize, outDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

        int fail = (opRet != 0) ? 1 : 0;
        if (fail == 0)
        {
            for (size_t i = 0; i < elemCount; ++i)
            {
                if (std::fabs(result[i] - expectedVals[i]) > tolerance)
                {
                    fail = 1;
                    break;
                }
            }
        }

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== ACL_DOUBLE test =====
    static int RunCumsumV2DoubleTest(const std::vector<int64_t> &shape,
                                     const std::vector<double> &inputVals,
                                     const std::vector<double> &expectedVals,
                                     int64_t dim, bool exclusive, bool reverse,
                                     double tolerance, aclrtStream stream)
    {
        size_t elemCount = ComputeElementCount(shape);
        if (elemCount == 0)
            return 0;
        size_t byteSize = elemCount * sizeof(double);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, inputVals.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_DOUBLE, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_DOUBLE, outDev);

        int opRet = RunCumsumV2Op(self, dim, exclusive, reverse, out, stream);

        std::vector<double> result(elemCount);
        aclrtMemcpy(result.data(), byteSize, outDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

        int fail = (opRet != 0) ? 1 : 0;
        if (fail == 0)
        {
            for (size_t i = 0; i < elemCount; ++i)
            {
                if (std::fabs(result[i] - expectedVals[i]) > tolerance)
                {
                    fail = 1;
                    break;
                }
            }
        }

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== ACL_FLOAT16 test =====
    static int RunCumsumV2Float16Test(const std::vector<int64_t> &shape,
                                      const std::vector<float> &inputVals,
                                      const std::vector<float> &expectedVals,
                                      int64_t dim, bool exclusive, bool reverse,
                                      float tolerance, aclrtStream stream)
    {
        size_t elemCount = ComputeElementCount(shape);
        if (elemCount == 0)
            return 0;
        std::vector<uint16_t> inputRaw(elemCount);
        for (size_t i = 0; i < elemCount; ++i)
        {
            inputRaw[i] = FloatToFloat16(inputVals[i]);
        }
        size_t byteSize = elemCount * sizeof(uint16_t);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, inputRaw.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_FLOAT16, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_FLOAT16, outDev);

        int opRet = RunCumsumV2Op(self, dim, exclusive, reverse, out, stream);

        std::vector<uint16_t> resultRaw(elemCount);
        aclrtMemcpy(resultRaw.data(), byteSize, outDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

        int fail = (opRet != 0) ? 1 : 0;
        if (fail == 0)
        {
            for (size_t i = 0; i < elemCount; ++i)
            {
                float resultFloat = Float16ToFloat(resultRaw[i]);
                if (std::fabs(resultFloat - expectedVals[i]) > tolerance)
                {
                    fail = 1;
                    break;
                }
            }
        }

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== ACL_BF16 test =====
    static int RunCumsumV2BF16Test(const std::vector<int64_t> &shape,
                                   const std::vector<float> &inputVals,
                                   const std::vector<float> &expectedVals,
                                   int64_t dim, bool exclusive, bool reverse,
                                   float tolerance, aclrtStream stream)
    {
        size_t elemCount = ComputeElementCount(shape);
        if (elemCount == 0)
            return 0;
        std::vector<uint16_t> inputRaw(elemCount);
        for (size_t i = 0; i < elemCount; ++i)
        {
            inputRaw[i] = FloatToBFloat16(inputVals[i]);
        }
        size_t byteSize = elemCount * sizeof(uint16_t);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, inputRaw.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_BF16, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_BF16, outDev);

        int opRet = RunCumsumV2Op(self, dim, exclusive, reverse, out, stream);

        std::vector<uint16_t> resultRaw(elemCount);
        aclrtMemcpy(resultRaw.data(), byteSize, outDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

        int fail = (opRet != 0) ? 1 : 0;
        if (fail == 0)
        {
            for (size_t i = 0; i < elemCount; ++i)
            {
                float resultFloat = BFloat16ToFloat(resultRaw[i]);
                if (std::fabs(resultFloat - expectedVals[i]) > tolerance)
                {
                    fail = 1;
                    break;
                }
            }
        }

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== ACL_INT32 test =====
    static int RunCumsumV2Int32Test(const std::vector<int64_t> &shape,
                                    const std::vector<int32_t> &inputVals,
                                    const std::vector<int32_t> &expectedVals,
                                    int64_t dim, bool exclusive, bool reverse,
                                    aclrtStream stream)
    {
        size_t elemCount = ComputeElementCount(shape);
        if (elemCount == 0)
            return 0;
        size_t byteSize = elemCount * sizeof(int32_t);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, inputVals.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_INT32, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_INT32, outDev);

        int opRet = RunCumsumV2Op(self, dim, exclusive, reverse, out, stream);

        std::vector<int32_t> result(elemCount);
        aclrtMemcpy(result.data(), byteSize, outDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

        int fail = (opRet != 0) ? 1 : 0;
        if (fail == 0)
        {
            for (size_t i = 0; i < elemCount; ++i)
            {
                if (result[i] != expectedVals[i])
                {
                    fail = 1;
                    break;
                }
            }
        }

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== ACL_INT64 test =====
    static int RunCumsumV2Int64Test(const std::vector<int64_t> &shape,
                                    const std::vector<int64_t> &inputVals,
                                    const std::vector<int64_t> &expectedVals,
                                    int64_t dim, bool exclusive, bool reverse,
                                    aclrtStream stream)
    {
        size_t elemCount = ComputeElementCount(shape);
        if (elemCount == 0)
            return 0;
        size_t byteSize = elemCount * sizeof(int64_t);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, inputVals.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_INT64, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_INT64, outDev);

        int opRet = RunCumsumV2Op(self, dim, exclusive, reverse, out, stream);

        std::vector<int64_t> result(elemCount);
        aclrtMemcpy(result.data(), byteSize, outDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

        int fail = (opRet != 0) ? 1 : 0;
        if (fail == 0)
        {
            for (size_t i = 0; i < elemCount; ++i)
            {
                if (result[i] != expectedVals[i])
                {
                    fail = 1;
                    break;
                }
            }
        }

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== ACL_INT8 test =====
    static int RunCumsumV2Int8Test(const std::vector<int64_t> &shape,
                                   const std::vector<int8_t> &inputVals,
                                   const std::vector<int8_t> &expectedVals,
                                   int64_t dim, bool exclusive, bool reverse,
                                   aclrtStream stream)
    {
        size_t elemCount = ComputeElementCount(shape);
        if (elemCount == 0)
            return 0;
        size_t byteSize = elemCount * sizeof(int8_t);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, inputVals.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_INT8, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_INT8, outDev);

        int opRet = RunCumsumV2Op(self, dim, exclusive, reverse, out, stream);

        std::vector<int8_t> result(elemCount);
        aclrtMemcpy(result.data(), byteSize, outDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

        int fail = (opRet != 0) ? 1 : 0;
        if (fail == 0)
        {
            for (size_t i = 0; i < elemCount; ++i)
            {
                if (result[i] != expectedVals[i])
                {
                    fail = 1;
                    break;
                }
            }
        }

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== ACL_INT16 test =====
    static int RunCumsumV2Int16Test(const std::vector<int64_t> &shape,
                                    const std::vector<int16_t> &inputVals,
                                    const std::vector<int16_t> &expectedVals,
                                    int64_t dim, bool exclusive, bool reverse,
                                    aclrtStream stream)
    {
        size_t elemCount = ComputeElementCount(shape);
        if (elemCount == 0)
            return 0;
        size_t byteSize = elemCount * sizeof(int16_t);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, inputVals.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_INT16, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_INT16, outDev);

        int opRet = RunCumsumV2Op(self, dim, exclusive, reverse, out, stream);

        std::vector<int16_t> result(elemCount);
        aclrtMemcpy(result.data(), byteSize, outDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

        int fail = (opRet != 0) ? 1 : 0;
        if (fail == 0)
        {
            for (size_t i = 0; i < elemCount; ++i)
            {
                if (result[i] != expectedVals[i])
                {
                    fail = 1;
                    break;
                }
            }
        }

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== ACL_UINT8 test =====
    static int RunCumsumV2Uint8Test(const std::vector<int64_t> &shape,
                                    const std::vector<uint8_t> &inputVals,
                                    const std::vector<uint8_t> &expectedVals,
                                    int64_t dim, bool exclusive, bool reverse,
                                    aclrtStream stream)
    {
        size_t elemCount = ComputeElementCount(shape);
        if (elemCount == 0)
            return 0;
        size_t byteSize = elemCount * sizeof(uint8_t);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, inputVals.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_UINT8, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_UINT8, outDev);

        int opRet = RunCumsumV2Op(self, dim, exclusive, reverse, out, stream);

        std::vector<uint8_t> result(elemCount);
        aclrtMemcpy(result.data(), byteSize, outDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

        int fail = (opRet != 0) ? 1 : 0;
        if (fail == 0)
        {
            for (size_t i = 0; i < elemCount; ++i)
            {
                if (result[i] != expectedVals[i])
                {
                    fail = 1;
                    break;
                }
            }
        }

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== ACL_COMPLEX64 test =====
    static int RunCumsumV2Complex64Test(const std::vector<int64_t> &shape,
                                        const std::vector<float> &inputInterleaved,
                                        const std::vector<float> &expectedInterleaved,
                                        int64_t dim, bool exclusive, bool reverse,
                                        float tolerance, aclrtStream stream)
    {
        size_t elemCount = ComputeElementCount(shape);
        if (elemCount == 0)
            return 0;
        size_t rawCount = elemCount * 2;
        size_t byteSize = rawCount * sizeof(float);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, inputInterleaved.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_COMPLEX64, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_COMPLEX64, outDev);

        int opRet = RunCumsumV2Op(self, dim, exclusive, reverse, out, stream);

        std::vector<float> resultInterleaved(rawCount);
        aclrtMemcpy(resultInterleaved.data(), byteSize, outDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

        int fail = (opRet != 0) ? 1 : 0;
        if (fail == 0)
        {
            for (size_t i = 0; i < rawCount; ++i)
            {
                if (std::fabs(resultInterleaved[i] - expectedInterleaved[i]) > tolerance)
                {
                    fail = 1;
                    break;
                }
            }
        }

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== ACL_COMPLEX128 test =====
    static int RunCumsumV2Complex128Test(const std::vector<int64_t> &shape,
                                         const std::vector<double> &inputInterleaved,
                                         const std::vector<double> &expectedInterleaved,
                                         int64_t dim, bool exclusive, bool reverse,
                                         double tolerance, aclrtStream stream)
    {
        size_t elemCount = ComputeElementCount(shape);
        if (elemCount == 0)
            return 0;
        size_t rawCount = elemCount * 2;
        size_t byteSize = rawCount * sizeof(double);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, inputInterleaved.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_COMPLEX128, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_COMPLEX128, outDev);

        int opRet = RunCumsumV2Op(self, dim, exclusive, reverse, out, stream);

        std::vector<double> resultInterleaved(rawCount);
        aclrtMemcpy(resultInterleaved.data(), byteSize, outDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

        int fail = (opRet != 0) ? 1 : 0;
        if (fail == 0)
        {
            for (size_t i = 0; i < rawCount; ++i)
            {
                if (std::fabs(resultInterleaved[i] - expectedInterleaved[i]) > tolerance)
                {
                    fail = 1;
                    break;
                }
            }
        }

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== 0D scalar tensor test =====
    static int TestCumsumV2Scalar(aclrtStream stream)
    {
        std::vector<int64_t> shape = {};
        float inputVal = 1.0f;
        float expectedVal = 1.0f;
        size_t byteSize = sizeof(float);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, &inputVal, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        std::vector<int64_t> strides = {};
        aclTensor *self = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
                                          ACL_FORMAT_ND, shape.data(), shape.size(), selfDev);
        aclTensor *out = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
                                         ACL_FORMAT_ND, shape.data(), shape.size(), outDev);

        int opRet = RunCumsumV2Op(self, 0, false, false, out, stream);

        float resultVal = 0.0f;
        aclrtMemcpy(&resultVal, byteSize, outDev, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

        int fail = (opRet != 0) ? 1 : 0;
        if (fail == 0 && std::fabs(resultVal - expectedVal) > 1e-6f)
        {
            fail = 1;
        }

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== Empty tensor test (shape with 0) =====
    static int TestCumsumV2Empty(aclrtStream stream)
    {
        std::vector<int64_t> shape = {2, 0};
        size_t minAlloc = 4; // minimal allocation for valid devPtr

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, minAlloc, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, minAlloc, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }

        std::vector<int64_t> strides = ComputeStrides(shape);
        aclTensor *self = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
                                          ACL_FORMAT_ND, shape.data(), shape.size(), selfDev);
        aclTensor *out = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
                                         ACL_FORMAT_ND, shape.data(), shape.size(), outDev);

        int opRet = RunCumsumV2Op(self, 1, false, false, out, stream);

        int fail = (opRet != 0) ? 1 : 0;

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== Error: null self =====
    static int TestErrorNullSelf(aclrtStream stream)
    {
        std::vector<int64_t> outShape = {2, 3};
        size_t outByteSize = 6 * sizeof(float);
        void *outDev = nullptr;
        if (aclrtMalloc(&outDev, outByteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;

        std::vector<int64_t> outStrides = ComputeStrides(outShape);
        aclTensor *out = aclCreateTensor(outShape.data(), outShape.size(), ACL_FLOAT, outStrides.data(),
                                         0, ACL_FORMAT_ND, outShape.data(), outShape.size(), outDev);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus ret = aclnnCumsumV2GetWorkspaceSize(nullptr, 1, false, false, out,
                                                        &workspaceSize, &executor);
        int fail = (ret == ACLNN_SUCCESS) ? 1 : 0;

        aclDestroyTensor(out);
        aclrtFree(outDev);
        return fail;
    }

    // ===== Error: null out =====
    static int TestErrorNullOut(aclrtStream stream)
    {
        std::vector<int64_t> selfShape = {2, 3};
        size_t selfByteSize = 6 * sizeof(float);
        std::vector<float> dummyData(6, 0.0f);
        void *selfDev = nullptr;
        if (aclrtMalloc(&selfDev, selfByteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        aclrtMemcpy(selfDev, selfByteSize, dummyData.data(), selfByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        std::vector<int64_t> selfStrides = ComputeStrides(selfShape);
        aclTensor *self = aclCreateTensor(selfShape.data(), selfShape.size(), ACL_FLOAT, selfStrides.data(),
                                          0, ACL_FORMAT_ND, selfShape.data(), selfShape.size(), selfDev);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus ret = aclnnCumsumV2GetWorkspaceSize(self, 1, false, false, nullptr,
                                                        &workspaceSize, &executor);
        int fail = (ret == ACLNN_SUCCESS) ? 1 : 0;

        aclDestroyTensor(self);
        aclrtFree(selfDev);
        return fail;
    }

    // ===== Error: dim out of range (positive) =====
    static int TestErrorDimOutOfRangePositive(aclrtStream stream)
    {
        std::vector<int64_t> shape = {2, 3};
        size_t byteSize = 6 * sizeof(float);
        std::vector<float> dummyData(6, 0.0f);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, dummyData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_FLOAT, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_FLOAT, outDev);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus ret = aclnnCumsumV2GetWorkspaceSize(self, 2, false, false, out,
                                                        &workspaceSize, &executor);
        int fail = (ret == ACLNN_SUCCESS) ? 1 : 0;

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== Error: dim out of range (negative) =====
    static int TestErrorDimOutOfRangeNegative(aclrtStream stream)
    {
        std::vector<int64_t> shape = {2, 3};
        size_t byteSize = 6 * sizeof(float);
        std::vector<float> dummyData(6, 0.0f);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, dummyData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_FLOAT, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_FLOAT, outDev);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus ret = aclnnCumsumV2GetWorkspaceSize(self, -3, false, false, out,
                                                        &workspaceSize, &executor);
        int fail = (ret == ACLNN_SUCCESS) ? 1 : 0;

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== Error: shape mismatch =====
    static int TestErrorShapeMismatch(aclrtStream stream)
    {
        std::vector<int64_t> selfShape = {2, 3};
        std::vector<int64_t> outShape = {3, 2};
        size_t byteSize = 6 * sizeof(float);
        std::vector<float> dummyData(6, 0.0f);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, dummyData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(selfShape, ACL_FLOAT, selfDev);
        aclTensor *out = CreateContiguousTensor(outShape, ACL_FLOAT, outDev);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus ret = aclnnCumsumV2GetWorkspaceSize(self, 1, false, false, out,
                                                        &workspaceSize, &executor);
        int fail = (ret == ACLNN_SUCCESS) ? 1 : 0;

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== Error: dtype mismatch =====
    static int TestErrorDtypeMismatch(aclrtStream stream)
    {
        std::vector<int64_t> shape = {2, 3};
        size_t selfByteSize = 6 * sizeof(float);
        size_t outByteSize = 6 * sizeof(int32_t);
        std::vector<float> dummyFloat(6, 0.0f);

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, selfByteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, outByteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, selfByteSize, dummyFloat.data(), selfByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_FLOAT, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_INT32, outDev);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus ret = aclnnCumsumV2GetWorkspaceSize(self, 1, false, false, out,
                                                        &workspaceSize, &executor);
        int fail = (ret == ACLNN_SUCCESS) ? 1 : 0;

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== Error: max dim exceeded (9D) =====
    static int TestErrorMaxDimExceeded(aclrtStream stream)
    {
        std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 1}; // 9 dimensions
        size_t byteSize = 1 * sizeof(float);
        float dummyVal = 0.0f;

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, &dummyVal, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_FLOAT, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_FLOAT, outDev);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus ret = aclnnCumsumV2GetWorkspaceSize(self, 1, false, false, out,
                                                        &workspaceSize, &executor);
        int fail = (ret == ACLNN_SUCCESS) ? 1 : 0;

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    // ===== Error: unsupported dtype (BOOL) =====
    static int TestErrorUnsupportedDtypeBool(aclrtStream stream)
    {
        std::vector<int64_t> shape = {2, 3};
        size_t byteSize = 6 * sizeof(uint8_t);
        std::vector<uint8_t> dummyData = {1, 0, 1, 0, 1, 1};

        void *selfDev = nullptr;
        void *outDev = nullptr;
        if (aclrtMalloc(&selfDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
            return 1;
        if (aclrtMalloc(&outDev, byteSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            aclrtFree(selfDev);
            return 1;
        }
        aclrtMemcpy(selfDev, byteSize, dummyData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclTensor *self = CreateContiguousTensor(shape, ACL_BOOL, selfDev);
        aclTensor *out = CreateContiguousTensor(shape, ACL_BOOL, outDev);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus ret = aclnnCumsumV2GetWorkspaceSize(self, 1, false, false, out,
                                                        &workspaceSize, &executor);
        int fail = (ret == ACLNN_SUCCESS) ? 1 : 0;

        aclDestroyTensor(self);
        aclDestroyTensor(out);
        aclrtFree(selfDev);
        aclrtFree(outDev);
        return fail;
    }

    int RunShard_cumsum_v2(aclrtStream stream)
    {
        int failCount = 0;

        // --- Normal scenarios (ACL_FLOAT) ---
        failCount += RunCumsumV2FloatTest({2, 3},
                                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                                          {1.0f, 3.0f, 6.0f, 4.0f, 9.0f, 15.0f},
                                          1, false, false, 1e-6f, stream);

        failCount += RunCumsumV2FloatTest({2, 3},
                                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                                          {0.0f, 1.0f, 3.0f, 0.0f, 4.0f, 9.0f},
                                          1, true, false, 1e-6f, stream);

        failCount += RunCumsumV2FloatTest({2, 3},
                                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                                          {6.0f, 5.0f, 3.0f, 15.0f, 11.0f, 6.0f},
                                          1, false, true, 1e-6f, stream);

        failCount += RunCumsumV2FloatTest({2, 3},
                                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                                          {5.0f, 3.0f, 0.0f, 11.0f, 6.0f, 0.0f},
                                          1, true, true, 1e-6f, stream);

        failCount += RunCumsumV2FloatTest({2, 3},
                                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                                          {1.0f, 3.0f, 6.0f, 4.0f, 9.0f, 15.0f},
                                          -1, false, false, 1e-6f, stream);

        failCount += RunCumsumV2FloatTest({2, 3},
                                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                                          {1.0f, 2.0f, 3.0f, 5.0f, 7.0f, 9.0f},
                                          0, false, false, 1e-6f, stream);

        failCount += RunCumsumV2FloatTest({5},
                                          {1.0f, -2.0f, 3.0f, 0.0f, 5.0f},
                                          {1.0f, -1.0f, 2.0f, 2.0f, 7.0f},
                                          0, false, false, 1e-6f, stream);

        failCount += TestCumsumV2Scalar(stream);
        failCount += TestCumsumV2Empty(stream);

        // --- Dtype scenarios ---
        failCount += RunCumsumV2FloatTest({3, 4},
                                          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                          {1.0f, 3.0f, 6.0f, 10.0f, 5.0f, 11.0f, 18.0f, 26.0f, 9.0f, 19.0f, 30.0f, 42.0f},
                                          1, false, false, 1e-6f, stream);

        failCount += RunCumsumV2Float16Test({3, 4},
                                            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                            {1.0f, 3.0f, 6.0f, 10.0f, 5.0f, 11.0f, 18.0f, 26.0f, 9.0f, 19.0f, 30.0f, 42.0f},
                                            1, false, false, 1e-2f, stream);

        failCount += RunCumsumV2BF16Test({3, 4},
                                         {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                         {1.0f, 3.0f, 6.0f, 10.0f, 5.0f, 11.0f, 18.0f, 26.0f, 9.0f, 19.0f, 30.0f, 42.0f},
                                         1, false, false, 1e-2f, stream);

        failCount += RunCumsumV2Int32Test({3, 4},
                                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                          {1, 3, 6, 10, 5, 11, 18, 26, 9, 19, 30, 42},
                                          1, false, false, stream);

        failCount += RunCumsumV2Int64Test({3, 4},
                                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                          {1, 3, 6, 10, 5, 11, 18, 26, 9, 19, 30, 42},
                                          1, false, false, stream);

        failCount += RunCumsumV2DoubleTest({3, 4},
                                           {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
                                           {1.0, 3.0, 6.0, 10.0, 5.0, 11.0, 18.0, 26.0, 9.0, 19.0, 30.0, 42.0},
                                           1, false, false, 1e-10, stream);

        failCount += RunCumsumV2Uint8Test({3, 4},
                                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                          {1, 3, 6, 10, 5, 11, 18, 26, 9, 19, 30, 42},
                                          1, false, false, stream);

        failCount += RunCumsumV2Int8Test({3, 4},
                                         {1, -2, 3, -4, 5, 6, -7, 8, -9, 10, 11, -12},
                                         {1, -1, 2, -2, 5, 11, 4, 12, -9, 1, 12, 0},
                                         1, false, false, stream);

        failCount += RunCumsumV2Int16Test({3, 4},
                                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                          {1, 3, 6, 10, 5, 11, 18, 26, 9, 19, 30, 42},
                                          1, false, false, stream);

        // cumsum_v2_dtype_complex64
        failCount += RunCumsumV2Complex64Test({3, 2},
                                              {1.0f, 0.0f, 2.0f, 1.0f, 3.0f, 0.0f, 4.0f, 2.0f, 5.0f, 0.0f, 6.0f, 3.0f},
                                              {1.0f, 0.0f, 3.0f, 1.0f, 3.0f, 0.0f, 7.0f, 2.0f, 5.0f, 0.0f, 11.0f, 3.0f},
                                              1, false, false, 1e-6f, stream);

        // cumsum_v2_dtype_complex128
        failCount += RunCumsumV2Complex128Test({3, 2},
                                               {1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 2.0, 5.0, 0.0, 6.0, 3.0},
                                               {1.0, 0.0, 3.0, 1.0, 3.0, 0.0, 7.0, 2.0, 5.0, 0.0, 11.0, 3.0},
                                               1, false, false, 1e-10, stream);

        // Error scenarios
        failCount += TestErrorNullSelf(stream);
        failCount += TestErrorNullOut(stream);
        failCount += TestErrorDimOutOfRangePositive(stream);
        failCount += TestErrorDimOutOfRangeNegative(stream);
        failCount += TestErrorShapeMismatch(stream);
        failCount += TestErrorDtypeMismatch(stream);
        failCount += TestErrorMaxDimExceeded(stream);
        failCount += TestErrorUnsupportedDtypeBool(stream);

        return failCount;
    }

} // namespace cumsum_v2

// ---------- ACL init / finalize ----------

static bool InitAclDevice(int32_t deviceId)
{
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "[ERROR] aclInit failed, ret = " << ret << std::endl;
        return false;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "[ERROR] aclrtSetDevice(" << deviceId << ") failed, ret = " << ret << std::endl;
        return false;
    }
    std::cout << "[INFO] ACL device " << deviceId << " initialized." << std::endl;
    return true;
}

static void FinalizeAclDevice(int32_t deviceId)
{
    aclError ret = aclrtResetDevice(deviceId);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "[WARN] aclrtResetDevice failed, ret = " << ret << std::endl;
    }
    ret = aclFinalize();
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "[WARN] aclFinalize failed, ret = " << ret << std::endl;
    }
    std::cout << "[INFO] ACL device " << deviceId << " finalized." << std::endl;
}

// ---------- main ----------

int32_t main(int32_t argc, char **argv)
{
    int32_t deviceId = 0;
    if (argc > 1)
    {
        deviceId = std::atoi(argv[1]);
    }

    if (!InitAclDevice(deviceId))
    {
        std::cerr << "[FATAL] ACL init failed, cannot run tests." << std::endl;
        return 1;
    }

    aclrtContext context = nullptr;
    aclError ctxRet = aclrtCreateContext(&context, deviceId);
    if (ctxRet != ACL_SUCCESS)
    {
        std::cerr << "[ERROR] aclrtCreateContext failed, ret = " << ctxRet << std::endl;
        FinalizeAclDevice(deviceId);
        return 1;
    }
    aclrtSetCurrentContext(context);

    aclrtStream stream = nullptr;
    aclError streamRet = aclrtCreateStream(&stream);
    if (streamRet != ACL_SUCCESS)
    {
        std::cerr << "[ERROR] aclrtCreateStream failed, ret = " << streamRet << std::endl;
        aclrtDestroyContext(context);
        FinalizeAclDevice(deviceId);
        return 1;
    }

    int32_t totalFailures = 0;

    // --- Run shard: cumsum_v1 ---
    int32_t fail1 = cumsum_v1::RunShard_cumsum_v1(stream);
    totalFailures += fail1;
    std::cout << "[INFO] Shard cumsum_v1 failures: " << fail1 << std::endl;

    // --- Run shard: cumsum_v2 ---
    int32_t fail2 = cumsum_v2::RunShard_cumsum_v2(stream);
    totalFailures += fail2;
    std::cout << "[INFO] Shard cumsum_v2 failures: " << fail2 << std::endl;

    // --- Aggregate result ---
    printf("\n========== SUMMARY ==========\n");
    std::cout << "[INFO] Total shard failures: " << totalFailures << std::endl;

    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    FinalizeAclDevice(deviceId);

    if (totalFailures > 0)
    {
        std::cerr << "[RESULT] Some tests FAILED (" << totalFailures << " total failures)." << std::endl;
        return totalFailures;
    }
    std::cout << "[RESULT] All tests PASSED." << std::endl;
    return 0;
}