/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_roll.h"

namespace {

template <typename T>
using DevicePtr = std::unique_ptr<void, std::function<void(void*)>>;

template <typename T>
using TensorPtr = std::unique_ptr<aclTensor, std::function<void(aclTensor*)>>;

using IntArrayPtr = std::unique_ptr<aclIntArray, std::function<void(const aclIntArray*)>>;

template <typename T>
struct RollCase {
    std::string name;
    std::vector<int64_t> view_shape;
    std::vector<int64_t> storage_shape;
    std::vector<int64_t> strides;
    int64_t offset = 0;
    std::vector<int64_t> shifts;
    std::vector<int64_t> dims;
    std::vector<T> storage_data;
    aclDataType dtype = ACL_FLOAT;
    double atol = 1e-6;
};

int64_t Numel(const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return 1;
    }
    return std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
}

std::vector<int64_t> MakeContiguousStrides(const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return {};
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

inline std::string FormatValue(uint8_t value)
{
    return std::to_string(static_cast<int>(value));
}

template <typename T>
std::string FormatValue(const T& value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

template <typename T>
std::string VectorToString(const std::vector<T>& values)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << FormatValue(values[i]);
    }
    oss << "]";
    return oss.str();
}

int64_t NormalizeShift(int64_t shift, int64_t dim_size)
{
    if (dim_size <= 0) {
        return 0;
    }
    int64_t normalized = shift % dim_size;
    if (normalized < 0) {
        normalized += dim_size;
    }
    return normalized;
}

int64_t CoordsToLinear(const std::vector<int64_t>& coords, const std::vector<int64_t>& strides)
{
    int64_t linear = 0;
    for (size_t i = 0; i < coords.size(); ++i) {
        linear += coords[i] * strides[i];
    }
    return linear;
}

std::vector<int64_t> LinearToCoords(int64_t linear, const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return {};
    }
    auto strides = MakeContiguousStrides(shape);
    std::vector<int64_t> coords(shape.size(), 0);
    for (size_t i = 0; i < shape.size(); ++i) {
        coords[i] = linear / strides[i];
        linear %= strides[i];
    }
    return coords;
}

template <typename T>
std::vector<T> MaterializeLogicalView(
    const std::vector<T>& storage,
    const std::vector<int64_t>& view_shape,
    const std::vector<int64_t>& storage_shape,
    const std::vector<int64_t>& strides,
    int64_t offset)
{
    const auto logical_size = Numel(view_shape);
    const auto actual_strides = strides.empty() ? MakeContiguousStrides(storage_shape.empty() ? view_shape : storage_shape) : strides;
    std::vector<T> logical(static_cast<size_t>(logical_size));
    for (int64_t linear = 0; linear < logical_size; ++linear) {
        const auto coords = LinearToCoords(linear, view_shape);
        int64_t storage_index = offset;
        for (size_t dim = 0; dim < coords.size(); ++dim) {
            storage_index += coords[dim] * actual_strides[dim];
        }
        logical[static_cast<size_t>(linear)] = storage[static_cast<size_t>(storage_index)];
    }
    return logical;
}

template <typename T>
std::vector<T> ReferenceRoll(
    const std::vector<T>& logical_input,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& shifts,
    const std::vector<int64_t>& dims)
{
    if (shape.empty()) {
        return logical_input;
    }

    const auto logical_size = Numel(shape);
    std::vector<T> output(static_cast<size_t>(logical_size));
    const auto logical_strides = MakeContiguousStrides(shape);

    if (dims.empty()) {
        const int64_t normalized_shift = NormalizeShift(shifts[0], logical_size);
        for (int64_t index = 0; index < logical_size; ++index) {
            const int64_t output_index = (index + normalized_shift) % logical_size;
            output[static_cast<size_t>(output_index)] = logical_input[static_cast<size_t>(index)];
        }
        return output;
    }

    for (int64_t out_linear = 0; out_linear < logical_size; ++out_linear) {
        auto in_coords = LinearToCoords(out_linear, shape);
        for (size_t i = 0; i < dims.size(); ++i) {
            int64_t dim = dims[i];
            if (dim < 0) {
                dim += static_cast<int64_t>(shape.size());
            }
            const int64_t dim_size = shape[static_cast<size_t>(dim)];
            const int64_t normalized_shift = NormalizeShift(shifts[i], dim_size);
            in_coords[static_cast<size_t>(dim)] =
                (in_coords[static_cast<size_t>(dim)] - normalized_shift + dim_size) % dim_size;
        }
        const int64_t in_linear = CoordsToLinear(in_coords, logical_strides);
        output[static_cast<size_t>(out_linear)] = logical_input[static_cast<size_t>(in_linear)];
    }
    return output;
}

template <typename T>
bool AreEqual(const T& actual, const T& expected, double atol)
{
    if (std::is_floating_point<T>::value) {
        return std::fabs(static_cast<double>(actual) - static_cast<double>(expected)) <= atol;
    }
    (void)atol;
    return actual == expected;
}

template <typename T>
void CheckVectorEqual(const std::vector<T>& actual, const std::vector<T>& expected, double atol, const std::string& name)
{
    if (actual.size() != expected.size()) {
        throw std::runtime_error(name + ": output size mismatch");
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!AreEqual(actual[i], expected[i], atol)) {
            std::ostringstream oss;
            oss << name << ": mismatch at index " << i << ", actual=";
            oss << FormatValue(actual[i]) << ", expected=" << FormatValue(expected[i]);
            throw std::runtime_error(oss.str());
        }
    }
}

class AclEnv {
public:
    AclEnv()
    {
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclInit failed");
        }
        initialized_ = true;

        ret = aclrtSetDevice(device_id_);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtSetDevice failed");
        }
        device_set_ = true;

        aclrtStream stream = nullptr;
        ret = aclrtCreateStream(&stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtCreateStream failed");
        }
        stream_ = stream;
    }

    ~AclEnv()
    {
        if (stream_ != nullptr) {
            (void)aclrtDestroyStream(stream_);
        }
        if (device_set_) {
            (void)aclrtResetDevice(device_id_);
        }
        if (initialized_) {
            (void)aclFinalize();
        }
    }

    aclrtStream stream() const
    {
        return stream_;
    }

private:
    int32_t device_id_ = 0;
    bool initialized_ = false;
    bool device_set_ = false;
    aclrtStream stream_ = nullptr;
};

template <typename T>
DevicePtr<T> MallocAndCopyToDevice(const std::vector<T>& host)
{
    void* device = nullptr;
    const size_t bytes = host.size() * sizeof(T);
    if (bytes > 0) {
        aclError ret = aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtMalloc failed");
        }
        ret = aclrtMemcpy(device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtMemcpy host to device failed");
        }
    }
    return DevicePtr<T>(device, [](void* p) {
        if (p != nullptr) {
            (void)aclrtFree(p);
        }
    });
}

template <typename T>
std::vector<T> CopyFromDevice(const void* device, size_t count)
{
    std::vector<T> host(count);
    const size_t bytes = count * sizeof(T);
    if (bytes > 0) {
        aclError ret = aclrtMemcpy(host.data(), bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtMemcpy device to host failed");
        }
    }
    return host;
}

template <typename T>
TensorPtr<T> CreateTensor(
    const std::vector<int64_t>& view_shape,
    const std::vector<int64_t>& strides,
    int64_t offset,
    aclFormat format,
    const std::vector<int64_t>& storage_shape,
    void* device_ptr,
    aclDataType dtype)
{
    aclTensor* tensor = aclCreateTensor(
        view_shape.empty() ? nullptr : view_shape.data(),
        view_shape.size(),
        dtype,
        strides.empty() ? nullptr : strides.data(),
        offset,
        format,
        storage_shape.empty() ? nullptr : storage_shape.data(),
        storage_shape.size(),
        device_ptr);
    if (tensor == nullptr) {
        throw std::runtime_error("aclCreateTensor failed");
    }
    return TensorPtr<T>(tensor, [](aclTensor* p) {
        if (p != nullptr) {
            aclDestroyTensor(p);
        }
    });
}

IntArrayPtr CreateIntArray(const std::vector<int64_t>& values)
{
    const int64_t* data = values.empty() ? nullptr : values.data();
    const aclIntArray* array = aclCreateIntArray(data, values.size());
    if (array == nullptr) {
        throw std::runtime_error("aclCreateIntArray failed");
    }
    return IntArrayPtr(const_cast<aclIntArray*>(array), [](const aclIntArray* p) {
        if (p != nullptr) {
            (void)aclDestroyIntArray(p);
        }
    });
}

template <typename T>
void RunRollCase(const RollCase<T>& test_case, aclrtStream stream)
{
    const auto storage_shape = test_case.storage_shape.empty() ? test_case.view_shape : test_case.storage_shape;
    const auto input_strides = test_case.strides.empty() ? MakeContiguousStrides(storage_shape) : test_case.strides;
    const auto output_strides = MakeContiguousStrides(test_case.view_shape);
    const auto output_elements = static_cast<size_t>(Numel(test_case.view_shape));
    std::vector<T> output_init(output_elements, T{});

    auto input_device = MallocAndCopyToDevice(test_case.storage_data);
    auto output_device = MallocAndCopyToDevice(output_init);

    auto input_tensor = CreateTensor<T>(
        test_case.view_shape, input_strides, test_case.offset, ACL_FORMAT_ND, storage_shape, input_device.get(), test_case.dtype);
    auto output_tensor = CreateTensor<T>(
        test_case.view_shape, output_strides, 0, ACL_FORMAT_ND, test_case.view_shape, output_device.get(), test_case.dtype);
    auto shifts = CreateIntArray(test_case.shifts);
    auto dims = CreateIntArray(test_case.dims);

    uint64_t workspace_size = 0;
    aclOpExecutor* raw_executor = nullptr;
    aclnnStatus status = aclnnRollGetWorkspaceSize(
        input_tensor.get(), shifts.get(), dims.get(), output_tensor.get(), &workspace_size, &raw_executor);
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(test_case.name + ": aclnnRollGetWorkspaceSize failed");
    }

    std::unique_ptr<void, std::function<void(void*)>> workspace(nullptr, [](void* p) {
        if (p != nullptr) {
            (void)aclrtFree(p);
        }
    });
    if (workspace_size > 0) {
        void* workspace_ptr = nullptr;
        aclError ret = aclrtMalloc(&workspace_ptr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(test_case.name + ": workspace allocation failed");
        }
        workspace.reset(workspace_ptr);
    }

    // Per the aclnn two-stage contract, phase-2 execution releases aclOpExecutor automatically.
    status = aclnnRoll(workspace.get(), workspace_size, raw_executor, stream);
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(test_case.name + ": aclnnRoll failed");
    }
    aclError ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(test_case.name + ": aclrtSynchronizeStream failed");
    }

    const auto actual = CopyFromDevice<T>(output_device.get(), output_elements);
    const auto logical_input = MaterializeLogicalView(
        test_case.storage_data, test_case.view_shape, storage_shape, input_strides, test_case.offset);
    const auto expected = ReferenceRoll(logical_input, test_case.view_shape, test_case.shifts, test_case.dims);
    CheckVectorEqual(actual, expected, test_case.atol, test_case.name);

    std::cout << "[ROLL_E2E] PASS " << test_case.name << " output=" << VectorToString(actual) << std::endl;
}

}  // namespace

int main()
{
    try {
        AclEnv env;

        RunRollCase<float>(
            {"float_axis0_contiguous",
             {4, 3},
             {4, 3},
             {},
             0,
             {1},
             {0},
             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
             ACL_FLOAT,
             1e-6},
            env.stream());

        RunRollCase<float>(
            {"float_axis1_contiguous",
             {4, 3},
             {4, 3},
             {},
             0,
             {1},
             {1},
             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
             ACL_FLOAT,
             1e-6},
            env.stream());

        RunRollCase<int32_t>(
            {"int32_multi_axis",
             {2, 3, 4},
             {2, 3, 4},
             {},
             0,
             {1, -2},
             {0, 2},
             {-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
             ACL_INT32,
             0.0},
            env.stream());

        RunRollCase<uint8_t>(
            {"bool_flatten_empty_dims",
             {2, 3},
             {2, 3},
             {},
             0,
             {2},
             {},
             {1, 0, 1, 0, 1, 0},
             ACL_BOOL,
             0.0},
            env.stream());

        RunRollCase<float>(
            {"float_non_contiguous",
             {5, 6},
             {6, 6},
             {1, 6},
             0,
             {2},
             {1},
             {0, 1, 2, 3, 4, 5,
              6, 7, 8, 9, 10, 11,
              12, 13, 14, 15, 16, 17,
              18, 19, 20, 21, 22, 23,
              24, 25, 26, 27, 28, 29,
              30, 31, 32, 33, 34, 35},
             ACL_FLOAT,
             1e-6},
            env.stream());

        RunRollCase<float>(
            {"float_zero_dim_tensor",
             {},
             {},
             {},
             0,
             {1},
             {},
             {5.0f},
             ACL_FLOAT,
             1e-6},
            env.stream());

        std::cout << "[ROLL_E2E] ALL PASSED" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[ROLL_E2E] FAIL " << ex.what() << std::endl;
        return 1;
    }
}
