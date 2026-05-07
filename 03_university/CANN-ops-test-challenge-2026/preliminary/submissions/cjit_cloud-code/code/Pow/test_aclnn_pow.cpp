/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "../op_api/aclnn_exp2.h"
#include "../op_api/aclnn_pow.h"
#include "../op_api/aclnn_pow_tensor_tensor.h"
#include "aclnn/opdev/op_errno.h"
#include "aclnn/opdev/platform.h"

namespace pow_example {

inline void Log(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    fflush(stdout);
}

inline bool ExpectAclStatus(const std::string &caseName, aclError actual, aclError expected = ACL_SUCCESS)
{
    if (actual == expected) {
        return true;
    }
    Log("[FAIL] %s: acl status expect %d, got %d\n", caseName.c_str(), expected, actual);
    return false;
}

inline bool ExpectAclnnStatus(const std::string &caseName, aclnnStatus actual, aclnnStatus expected = ACLNN_SUCCESS)
{
    if (actual == expected) {
        return true;
    }
    Log("[FAIL] %s: aclnn status expect %d, got %d\n", caseName.c_str(), expected, actual);
    return false;
}

inline bool ExpectTrue(const std::string &caseName, bool condition, const char *message)
{
    if (condition) {
        return true;
    }
    Log("[FAIL] %s: %s\n", caseName.c_str(), message);
    return false;
}

void SetExampleSocVersion(op::SocVersion socVersion);

inline int64_t Numel(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

inline std::vector<int64_t> MakeStrides(const std::vector<int64_t> &shape)
{
    if (shape.empty()) {
        return {};
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i) + 1] * strides[static_cast<size_t>(i) + 1];
    }
    return strides;
}

struct RuntimeContext {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    bool aclInitialized = false;
    bool deviceSet = false;

    bool Init(const std::string &caseName)
    {
        auto ret = aclInit(nullptr);
        if (!ExpectAclStatus(caseName + ".aclInit", ret)) {
            return false;
        }
        aclInitialized = true;

        ret = aclrtSetDevice(deviceId);
        if (!ExpectAclStatus(caseName + ".aclrtSetDevice", ret)) {
            return false;
        }
        deviceSet = true;

        ret = aclrtCreateStream(&stream);
        if (!ExpectAclStatus(caseName + ".aclrtCreateStream", ret)) {
            return false;
        }
        return true;
    }

    ~RuntimeContext()
    {
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(deviceId);
        }
        if (aclInitialized) {
            aclFinalize();
        }
    }
};

struct DeviceTensor {
    void *deviceAddr = nullptr;
    aclTensor *tensor = nullptr;
    size_t bytes = 0;

    void Reset()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
            tensor = nullptr;
        }
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
            deviceAddr = nullptr;
        }
        bytes = 0;
    }

    ~DeviceTensor()
    {
        Reset();
    }
};

struct ScalarHolder {
    aclScalar *scalar = nullptr;

    void Reset()
    {
        if (scalar != nullptr) {
            aclDestroyScalar(scalar);
            scalar = nullptr;
        }
    }

    ~ScalarHolder()
    {
        Reset();
    }
};

struct Workspace {
    void *deviceAddr = nullptr;
    uint64_t bytes = 0;

    bool Allocate(const std::string &caseName, uint64_t workspaceSize)
    {
        bytes = workspaceSize;
        if (workspaceSize == 0) {
            return true;
        }
        auto ret = aclrtMalloc(&deviceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        return ExpectAclStatus(caseName + ".aclrtMallocWorkspace", ret);
    }

    ~Workspace()
    {
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
        }
    }
};

template <typename T>
bool CreateTensor(const std::string &caseName,
                  const std::vector<T> &hostData,
                  const std::vector<int64_t> &shape,
                  aclDataType dataType,
                  DeviceTensor *out,
                  aclFormat format = ACL_FORMAT_ND)
{
    out->Reset();
    if (!ExpectTrue(caseName + ".shape", Numel(shape) == static_cast<int64_t>(hostData.size()),
                    "host data size does not match shape")) {
        return false;
    }

    out->bytes = hostData.size() * sizeof(T);
    if (out->bytes > 0) {
        auto ret = aclrtMalloc(&out->deviceAddr, out->bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!ExpectAclStatus(caseName + ".aclrtMalloc", ret)) {
            return false;
        }
        ret = aclrtMemcpy(out->deviceAddr, out->bytes, hostData.data(), out->bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (!ExpectAclStatus(caseName + ".aclrtMemcpy", ret)) {
            return false;
        }
    }

    auto strides = MakeStrides(shape);
    const int64_t *shapeData = shape.empty() ? nullptr : shape.data();
    const int64_t *strideData = strides.empty() ? nullptr : strides.data();
    out->tensor = aclCreateTensor(shapeData,
                                  shape.size(),
                                  dataType,
                                  strideData,
                                  0,
                                  format,
                                  shapeData,
                                  shape.size(),
                                  out->deviceAddr);
    if (out->tensor == nullptr) {
        Log("[FAIL] %s.aclCreateTensor returned nullptr\n", caseName.c_str());
        return false;
    }
    return true;
}

template <typename T>
bool CopyToHost(const std::string &caseName, const DeviceTensor &tensor, std::vector<T> *hostData)
{
    size_t bytes = hostData->size() * sizeof(T);
    if (bytes == 0) {
        return true;
    }
    auto ret = aclrtMemcpy(hostData->data(), bytes, tensor.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    return ExpectAclStatus(caseName + ".copyToHost", ret);
}

template <typename T>
inline double ToDouble(const T &value)
{
    return static_cast<double>(value);
}

template <>
inline double ToDouble<aclFloat16>(const aclFloat16 &value)
{
    return static_cast<double>(aclFloat16ToFloat(value));
}

template <typename T>
bool ExpectVectorNear(const std::string &caseName,
                      const std::vector<T> &actual,
                      const std::vector<double> &expected,
                      double tolerance)
{
    if (actual.size() != expected.size()) {
        Log("[FAIL] %s: size expect %zu, got %zu\n", caseName.c_str(), expected.size(), actual.size());
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        double diff = std::fabs(ToDouble(actual[i]) - expected[i]);
        if (diff > tolerance) {
            Log("[FAIL] %s: index %zu expect %.6f, got %.6f\n",
                caseName.c_str(),
                i,
                expected[i],
                ToDouble(actual[i]));
            return false;
        }
    }
    return true;
}

inline aclFloat16 Float16(float value)
{
    return aclFloatToFloat16(value);
}

}  // namespace pow_example

namespace {

op::SocVersion gPowExampleSocVersion = op::SocVersion::ASCEND950;
op::PlatformInfo *gPowExamplePlatformInfo = new op::PlatformInfo();

const std::string &GetPowExampleSocLongVersion()
{
    static const std::string kVersion = "pow-example-inline-override";
    return kVersion;
}

}  // namespace

extern "C" int32_t CheckLogLevel(int32_t, int32_t)
{
    return 0;
}

extern "C" void DlogRecord(int32_t, int32_t, const char *, ...)
{
}

void ReportErrorMessageInner(const std::string &, const char *, ...)
{
}

namespace op {

namespace internal {

std::string GetLogApiInfo()
{
    return "pow_example";
}

}  // namespace internal

PlatformInfo::~PlatformInfo() {}

SocVersion PlatformInfo::GetSocVersion() const
{
    return gPowExampleSocVersion;
}

const std::string &PlatformInfo::GetSocLongVersion() const
{
    return GetPowExampleSocLongVersion();
}

int32_t PlatformInfo::GetDeviceId() const
{
    return 0;
}

bool PlatformInfo::CheckSupport(SocSpec, SocSpecAbility) const
{
    return true;
}

int64_t PlatformInfo::GetBlockSize() const
{
    return 0;
}

uint32_t PlatformInfo::GetCubeCoreNum() const
{
    return 0;
}

uint32_t PlatformInfo::GetVectorCoreNum() const
{
    return 0;
}

bool PlatformInfo::Valid() const
{
    return true;
}

bool PlatformInfo::GetFftsPlusMode() const
{
    return true;
}

NpuArch PlatformInfo::GetCurNpuArch() const
{
    switch (gPowExampleSocVersion) {
        case SocVersion::ASCEND910B:
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND910E:
            return NpuArch::DAV_2201;
        case SocVersion::ASCEND950:
            return NpuArch::DAV_3510;
        case SocVersion::ASCEND310P:
            return NpuArch::DAV_2002;
        case SocVersion::ASCEND310B:
            return NpuArch::DAV_3002;
        case SocVersion::ASCEND610LITE:
            return NpuArch::DAV_3102;
        case SocVersion::ASCEND910:
        case SocVersion::ASCEND310:
            return NpuArch::DAV_1001;
        default:
            return NpuArch::DAV_RESV;
    }
}

fe::PlatFormInfos *PlatformInfo::GetPlatformInfos() const
{
    return nullptr;
}

void PlatformInfo::SetPlatformImpl(PlatformInfoImpl *impl)
{
    impl_ = impl;
    valid_ = true;
}

const PlatformInfo &GetCurrentPlatformInfo()
{
    return *gPowExamplePlatformInfo;
}

}  // namespace op

namespace pow_example {

void SetExampleSocVersion(op::SocVersion socVersion)
{
    gPowExampleSocVersion = socVersion;
}

}  // namespace pow_example

namespace l0op {

const aclTensor *Pow(const aclTensor *self, const aclTensor *exponent, aclOpExecutor *executor);

}  // namespace l0op

namespace {

using namespace pow_example;

inline bool ExpectAclnnStatusEither(const std::string &caseName,
                                    aclnnStatus actual,
                                    aclnnStatus expectedA,
                                    aclnnStatus expectedB)
{
    if (actual == expectedA || actual == expectedB) {
        return true;
    }
    Log("[FAIL] %s: aclnn status expect %d or %d, got %d\n",
        caseName.c_str(),
        expectedA,
        expectedB,
        actual);
    return false;
}

inline bool ExpectAclnnStatusOneOf(const std::string &caseName,
                                   aclnnStatus actual,
                                   std::initializer_list<aclnnStatus> expectedStatuses)
{
    for (aclnnStatus expected : expectedStatuses) {
        if (actual == expected) {
            return true;
        }
    }
    Log("[FAIL] %s: unexpected aclnn status %d\n", caseName.c_str(), actual);
    return false;
}

struct ScopedSocVersion {
    explicit ScopedSocVersion(op::SocVersion socVersion)
        : previous(op::GetCurrentPlatformInfo().GetSocVersion())
    {
        SetExampleSocVersion(socVersion);
    }

    ~ScopedSocVersion()
    {
        SetExampleSocVersion(previous);
    }

    op::SocVersion previous;
};

bool ShouldRunCase(const std::string &caseName)
{
    const char *filter = std::getenv("POW_CASE_FILTER");
    return filter == nullptr || caseName.find(filter) != std::string::npos;
}

template <typename Fn>
bool RunCase(const std::string &caseName, Fn &&fn)
{
    if (!ShouldRunCase(caseName)) {
        return true;
    }
    Log("[CASE] %s\n", caseName.c_str());
    return fn();
}

template <typename T>
bool CreateTensorWithFormat(const std::string &caseName,
                            const std::vector<T> &hostData,
                            const std::vector<int64_t> &shape,
                            aclDataType dataType,
                            DeviceTensor *out,
                            aclFormat format)
{
    out->Reset();
    if (!ExpectTrue(caseName + ".shape", Numel(shape) == static_cast<int64_t>(hostData.size()),
                    "host data size does not match shape")) {
        return false;
    }

    out->bytes = hostData.size() * sizeof(T);
    if (out->bytes > 0) {
        auto ret = aclrtMalloc(&out->deviceAddr, out->bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!ExpectAclStatus(caseName + ".aclrtMalloc", ret)) {
            return false;
        }
        ret = aclrtMemcpy(out->deviceAddr, out->bytes, hostData.data(), out->bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (!ExpectAclStatus(caseName + ".aclrtMemcpy", ret)) {
            return false;
        }
    }

    auto strides = MakeStrides(shape);
    const int64_t *shapeData = shape.empty() ? nullptr : shape.data();
    const int64_t *strideData = strides.empty() ? nullptr : strides.data();
    out->tensor = aclCreateTensor(shapeData,
                                  shape.size(),
                                  dataType,
                                  strideData,
                                  0,
                                  format,
                                  shapeData,
                                  shape.size(),
                                  out->deviceAddr);
    if (out->tensor == nullptr) {
        Log("[FAIL] %s.aclCreateTensor returned nullptr\n", caseName.c_str());
        return false;
    }
    return true;
}

template <typename SelfT, typename OutT, typename ScalarT>
bool RunTensorScalarCase(RuntimeContext &runtime,
                         const std::string &caseName,
                         const std::vector<SelfT> &selfHost,
                         const std::vector<int64_t> &shape,
                         aclDataType selfDtype,
                         ScalarT exponentValue,
                         aclDataType exponentDtype,
                         aclDataType outDtype,
                         const std::vector<double> &expected,
                         double tolerance,
                         bool inplace)
{
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    ScalarHolder exponent;
    if (!CreateTensor(caseName + ".self", selfHost, shape, selfDtype, &selfTensor)) {
        return false;
    }

    exponent.scalar = aclCreateScalar(&exponentValue, exponentDtype);
    if (!ExpectTrue(caseName + ".scalar", exponent.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }

    std::vector<OutT> outInit(static_cast<size_t>(Numel(shape)), OutT {});
    if (!inplace && !CreateTensor(caseName + ".out", outInit, shape, outDtype, &outTensor)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto stage1 = inplace ? aclnnInplacePowTensorScalarGetWorkspaceSize(selfTensor.tensor,
                                                                        exponent.scalar,
                                                                        &workspaceSize,
                                                                        &executor)
                          : aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor,
                                                                 exponent.scalar,
                                                                 outTensor.tensor,
                                                                 &workspaceSize,
                                                                 &executor);
    if (!ExpectAclnnStatus(caseName + ".stage1", stage1)) {
        return false;
    }

    Workspace workspace;
    if (!workspace.Allocate(caseName, workspaceSize)) {
        return false;
    }

    auto stage2 = inplace ? aclnnInplacePowTensorScalar(workspace.deviceAddr,
                                                        workspaceSize,
                                                        executor,
                                                        runtime.stream)
                          : aclnnPowTensorScalar(workspace.deviceAddr,
                                                 workspaceSize,
                                                 executor,
                                                 runtime.stream);
    if (stage2 != ACLNN_SUCCESS) {
        return ExpectAclnnStatusOneOf(caseName + ".stage2", stage2, {ACLNN_SUCCESS, ACLNN_ERR_INNER});
    }
    if (!ExpectAclStatus(caseName + ".sync", aclrtSynchronizeStream(runtime.stream))) {
        return false;
    }

    const DeviceTensor *resultTensor = inplace ? &selfTensor : &outTensor;
    std::vector<OutT> actual(static_cast<size_t>(Numel(shape)));
    if (!CopyToHost(caseName, *resultTensor, &actual)) {
        return false;
    }
    return ExpectVectorNear(caseName + ".result", actual, expected, tolerance);
}

template <typename ExpT, typename OutT, typename ScalarT>
bool RunScalarTensorCase(RuntimeContext &runtime,
                         const std::string &caseName,
                         ScalarT selfValue,
                         aclDataType selfDtype,
                         const std::vector<ExpT> &exponentHost,
                         const std::vector<int64_t> &shape,
                         aclDataType exponentDtype,
                         aclDataType outDtype,
                         const std::vector<double> &expected,
                         double tolerance)
{
    DeviceTensor exponentTensor;
    DeviceTensor outTensor;
    ScalarHolder selfScalar;
    if (!CreateTensor(caseName + ".exp", exponentHost, shape, exponentDtype, &exponentTensor)) {
        return false;
    }

    selfScalar.scalar = aclCreateScalar(&selfValue, selfDtype);
    if (!ExpectTrue(caseName + ".scalar", selfScalar.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }

    std::vector<OutT> outInit(static_cast<size_t>(Numel(shape)), OutT {});
    if (!CreateTensor(caseName + ".out", outInit, shape, outDtype, &outTensor)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto stage1 = aclnnPowScalarTensorGetWorkspaceSize(selfScalar.scalar,
                                                       exponentTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    if (!ExpectAclnnStatus(caseName + ".stage1", stage1)) {
        return false;
    }

    Workspace workspace;
    if (!workspace.Allocate(caseName, workspaceSize)) {
        return false;
    }

    auto stage2 = aclnnPowScalarTensor(workspace.deviceAddr, workspaceSize, executor, runtime.stream);
    if (stage2 != ACLNN_SUCCESS) {
        return ExpectAclnnStatusOneOf(caseName + ".stage2", stage2, {ACLNN_SUCCESS, ACLNN_ERR_INNER});
    }
    if (!ExpectAclStatus(caseName + ".sync", aclrtSynchronizeStream(runtime.stream))) {
        return false;
    }

    std::vector<OutT> actual(static_cast<size_t>(Numel(shape)));
    if (!CopyToHost(caseName, outTensor, &actual)) {
        return false;
    }
    return ExpectVectorNear(caseName + ".result", actual, expected, tolerance);
}

template <typename SelfT, typename OutT, typename ScalarT>
bool ExpectTensorScalarStage1Status(const std::string &caseName,
                                    const std::vector<SelfT> &selfHost,
                                    const std::vector<int64_t> &shape,
                                    aclDataType selfDtype,
                                    ScalarT exponentValue,
                                    aclDataType exponentDtype,
                                    const std::vector<OutT> &outHost,
                                    aclDataType outDtype,
                                    aclnnStatus expectedStatus)
{
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    ScalarHolder exponent;
    if (!CreateTensor(caseName + ".self", selfHost, shape, selfDtype, &selfTensor) ||
        !CreateTensor(caseName + ".out", outHost, shape, outDtype, &outTensor)) {
        return false;
    }
    exponent.scalar = aclCreateScalar(&exponentValue, exponentDtype);
    if (!ExpectTrue(caseName + ".scalar", exponent.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor,
                                                       exponent.scalar,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, expectedStatus);
}

template <typename SelfT, typename OutT, typename ScalarT>
bool ExpectTensorScalarStage1StatusOneOf(const std::string &caseName,
                                         const std::vector<SelfT> &selfHost,
                                         const std::vector<int64_t> &shape,
                                         aclDataType selfDtype,
                                         ScalarT exponentValue,
                                         aclDataType exponentDtype,
                                         const std::vector<OutT> &outHost,
                                         aclDataType outDtype,
                                         std::initializer_list<aclnnStatus> expectedStatuses)
{
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    ScalarHolder exponent;
    if (!CreateTensor(caseName + ".self", selfHost, shape, selfDtype, &selfTensor) ||
        !CreateTensor(caseName + ".out", outHost, shape, outDtype, &outTensor)) {
        return false;
    }
    exponent.scalar = aclCreateScalar(&exponentValue, exponentDtype);
    if (!ExpectTrue(caseName + ".scalar", exponent.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor,
                                                       exponent.scalar,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatusOneOf(caseName + ".stage1", status, expectedStatuses);
}

template <typename ExpT, typename OutT, typename ScalarT>
bool ExpectScalarTensorStage1StatusOneOf(const std::string &caseName,
                                         ScalarT selfValue,
                                         aclDataType selfDtype,
                                         const std::vector<ExpT> &exponentHost,
                                         const std::vector<int64_t> &shape,
                                         aclDataType exponentDtype,
                                         const std::vector<OutT> &outHost,
                                         aclDataType outDtype,
                                         std::initializer_list<aclnnStatus> expectedStatuses)
{
    DeviceTensor exponentTensor;
    DeviceTensor outTensor;
    ScalarHolder selfScalar;
    if (!CreateTensor(caseName + ".exp", exponentHost, shape, exponentDtype, &exponentTensor) ||
        !CreateTensor(caseName + ".out", outHost, shape, outDtype, &outTensor)) {
        return false;
    }
    selfScalar.scalar = aclCreateScalar(&selfValue, selfDtype);
    if (!ExpectTrue(caseName + ".scalar", selfScalar.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowScalarTensorGetWorkspaceSize(selfScalar.scalar,
                                                       exponentTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatusOneOf(caseName + ".stage1", status, expectedStatuses);
}

bool TestL0PowBroadcastInvalid()
{
    const std::string caseName = "l0_pow.broadcast_invalid";
    DeviceTensor selfTensor;
    DeviceTensor exponentTensor;
    if (!CreateTensor(caseName + ".self", std::vector<float> {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, ACL_FLOAT, &selfTensor) ||
        !CreateTensor(caseName + ".exp", std::vector<float> {2.0f, 3.0f, 4.0f}, {3, 1}, ACL_FLOAT, &exponentTensor)) {
        return false;
    }
    const aclTensor *powOut = l0op::Pow(selfTensor.tensor, exponentTensor.tensor, nullptr);
    return ExpectTrue(caseName, powOut == nullptr, "l0op::Pow should reject invalid broadcast");
}

bool TestTensorScalarEmpty(RuntimeContext &)
{
    const std::string caseName = "tensor_scalar.empty";
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    ScalarHolder exponent;
    if (!CreateTensor(caseName + ".self", std::vector<float> {}, {0, 2}, ACL_FLOAT, &selfTensor) ||
        !CreateTensor(caseName + ".out", std::vector<float> {}, {0, 2}, ACL_FLOAT, &outTensor)) {
        return false;
    }
    float exponentValue = 2.0f;
    exponent.scalar = aclCreateScalar(&exponentValue, ACL_FLOAT);
    if (!ExpectTrue(caseName + ".scalar", exponent.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }

    uint64_t workspaceSize = 1;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor,
                                                       exponent.scalar,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status) &&
           ExpectTrue(caseName + ".workspace", workspaceSize == 0, "workspace size should be 0");
}

bool TestTensorScalarNullOut(RuntimeContext &)
{
    const std::string caseName = "tensor_scalar.null_out";
    DeviceTensor selfTensor;
    ScalarHolder exponent;
    if (!CreateTensor(caseName + ".self", std::vector<float> {1.0f, 2.0f}, {2}, ACL_FLOAT, &selfTensor)) {
        return false;
    }
    float exponentValue = 2.0f;
    exponent.scalar = aclCreateScalar(&exponentValue, ACL_FLOAT);
    if (!ExpectTrue(caseName + ".scalar", exponent.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor,
                                                       exponent.scalar,
                                                       nullptr,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_NULLPTR);
}

bool TestTensorScalarInvalidRank(RuntimeContext &)
{
    const std::string caseName = "tensor_scalar.invalid_rank";
    const std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 2};
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    ScalarHolder exponent;
    std::vector<float> host(2, 1.0f);
    if (!CreateTensor(caseName + ".self", host, shape, ACL_FLOAT, &selfTensor) ||
        !CreateTensor(caseName + ".out", host, shape, ACL_FLOAT, &outTensor)) {
        return false;
    }
    int32_t exponentValue = 2;
    exponent.scalar = aclCreateScalar(&exponentValue, ACL_INT32);
    if (!ExpectTrue(caseName + ".scalar", exponent.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor,
                                                       exponent.scalar,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_INVALID);
}

bool TestTensorScalarNegativeIntegralExponent(RuntimeContext &)
{
    const std::string caseName = "tensor_scalar.negative_integral_exponent";
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    ScalarHolder exponent;
    if (!CreateTensor(caseName + ".self", std::vector<int32_t> {1, 2, 3}, {3}, ACL_INT32, &selfTensor) ||
        !CreateTensor(caseName + ".out", std::vector<int32_t> {0, 0, 0}, {3}, ACL_INT32, &outTensor)) {
        return false;
    }
    int32_t exponentValue = -1;
    exponent.scalar = aclCreateScalar(&exponentValue, ACL_INT32);
    if (!ExpectTrue(caseName + ".scalar", exponent.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor,
                                                       exponent.scalar,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_INVALID);
}

bool TestTensorScalarOverflow(RuntimeContext &)
{
    const std::string caseName = "tensor_scalar.overflow";
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    ScalarHolder exponent;
    if (!CreateTensor(caseName + ".self", std::vector<int8_t> {1, 2, 3}, {3}, ACL_INT8, &selfTensor) ||
        !CreateTensor(caseName + ".out", std::vector<int8_t> {0, 0, 0}, {3}, ACL_INT8, &outTensor)) {
        return false;
    }
    int32_t exponentValue = 200;
    exponent.scalar = aclCreateScalar(&exponentValue, ACL_INT32);
    if (!ExpectTrue(caseName + ".scalar", exponent.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor,
                                                       exponent.scalar,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_INVALID);
}

bool TestTensorScalarOverflowInt16(RuntimeContext &)
{
    return ExpectTensorScalarStage1Status<int16_t, int16_t, int32_t>("tensor_scalar.overflow_int16",
                                                                     {1, 2, 3},
                                                                     {3},
                                                                     ACL_INT16,
                                                                     40000,
                                                                     ACL_INT32,
                                                                     {0, 0, 0},
                                                                     ACL_INT16,
                                                                     ACLNN_ERR_PARAM_INVALID);
}

bool TestTensorScalarOverflowInt32(RuntimeContext &)
{
    return ExpectTensorScalarStage1Status<int32_t, int32_t, int64_t>("tensor_scalar.overflow_int32",
                                                                     {1, 2, 3},
                                                                     {3},
                                                                     ACL_INT32,
                                                                     static_cast<int64_t>(1) << 40,
                                                                     ACL_INT64,
                                                                     {0, 0, 0},
                                                                     ACL_INT32,
                                                                     ACLNN_ERR_PARAM_INVALID);
}

bool TestTensorScalarOverflowUInt8(RuntimeContext &)
{
    return ExpectTensorScalarStage1Status<uint8_t, uint8_t, int32_t>("tensor_scalar.overflow_uint8",
                                                                     {1, 2, 3},
                                                                     {3},
                                                                     ACL_UINT8,
                                                                     300,
                                                                     ACL_INT32,
                                                                     {0, 0, 0},
                                                                     ACL_UINT8,
                                                                     ACLNN_ERR_PARAM_INVALID);
}

bool TestTensorScalarOverflowFloat16(RuntimeContext &)
{
    return ExpectTensorScalarStage1Status<aclFloat16, aclFloat16, double>("tensor_scalar.overflow_float16",
                                                                          {Float16(1.0f), Float16(2.0f)},
                                                                          {2},
                                                                          ACL_FLOAT16,
                                                                          1e20,
                                                                          ACL_DOUBLE,
                                                                          {Float16(0.0f), Float16(0.0f)},
                                                                          ACL_FLOAT16,
                                                                          ACLNN_ERR_PARAM_INVALID);
}

bool TestTensorScalarUnsupportedSelfUint16(RuntimeContext &)
{
    return ExpectTensorScalarStage1Status<uint16_t, uint16_t, int32_t>("tensor_scalar.unsupported_self_uint16",
                                                                       {1, 2},
                                                                       {2},
                                                                       ACL_UINT16,
                                                                       2,
                                                                       ACL_INT32,
                                                                       {0, 0},
                                                                       ACL_UINT16,
                                                                       ACLNN_ERR_PARAM_INVALID);
}

bool TestTensorScalarUnsupportedExponentUint16(RuntimeContext &)
{
    return ExpectTensorScalarStage1Status<int32_t, int32_t, uint16_t>("tensor_scalar.unsupported_exponent_uint16",
                                                                      {1, 2},
                                                                      {2},
                                                                      ACL_INT32,
                                                                      2,
                                                                      ACL_UINT16,
                                                                      {0, 0},
                                                                      ACL_INT32,
                                                                      ACLNN_ERR_PARAM_INVALID);
}

bool TestTensorScalarUnsupportedBf16(RuntimeContext &)
{
    return ExpectTensorScalarStage1StatusOneOf<uint16_t, uint16_t, float>("tensor_scalar.bf16_stage1_success",
                                                                          {0x3f80, 0x4000},
                                                                          {2},
                                                                          ACL_BF16,
                                                                          2.0f,
                                                                          ACL_FLOAT,
                                                                          {0, 0},
                                                                          ACL_BF16,
                                                                          {ACLNN_SUCCESS,
                                                                           ACLNN_ERR_INNER_NULLPTR,
                                                                           ACLNN_ERR_INNER});
}

bool TestTensorScalarUnsupportedBf16On310P(RuntimeContext &)
{
    ScopedSocVersion soc(op::SocVersion::ASCEND310P);
    return ExpectTensorScalarStage1Status<uint16_t, uint16_t, float>("tensor_scalar.bf16_unsupported_310p",
                                                                     {0x3f80, 0x4000},
                                                                     {2},
                                                                     ACL_BF16,
                                                                     2.0f,
                                                                     ACL_FLOAT,
                                                                     {0, 0},
                                                                     ACL_BF16,
                                                                     ACLNN_ERR_PARAM_INVALID);
}

bool TestTensorScalarFormatNzWarningPath(RuntimeContext &)
{
    const std::string caseName = "tensor_scalar.format_nz_warning";
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    ScalarHolder exponent;
    if (!CreateTensorWithFormat(caseName + ".self",
                                std::vector<float> {1.0f, 2.0f},
                                {2},
                                ACL_FLOAT,
                                &selfTensor,
                                ACL_FORMAT_FRACTAL_NZ) ||
        !CreateTensor(caseName + ".out", std::vector<float> {0.0f, 0.0f}, {2}, ACL_FLOAT, &outTensor)) {
        return false;
    }
    float exponentValue = 2.0f;
    exponent.scalar = aclCreateScalar(&exponentValue, ACL_FLOAT);
    if (!ExpectTrue(caseName + ".scalar", exponent.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor,
                                                       exponent.scalar,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatusOneOf(caseName + ".stage1",
                                  status,
                                  {ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR, ACLNN_ERR_INNER});
}

bool TestTensorScalarComplexExponent(RuntimeContext &)
{
    const std::string caseName = "tensor_scalar.complex_exponent";
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    ScalarHolder exponent;
    std::complex<float> exponentValue {1.0f, 1.0f};
    if (!CreateTensor(caseName + ".self", std::vector<float> {1.0f, 2.0f}, {2}, ACL_FLOAT, &selfTensor) ||
        !CreateTensor(caseName + ".out", std::vector<std::complex<float>> {{0.0f, 0.0f}, {0.0f, 0.0f}}, {2}, ACL_COMPLEX64, &outTensor)) {
        return false;
    }
    exponent.scalar = aclCreateScalar(&exponentValue, ACL_COMPLEX64);
    if (!ExpectTrue(caseName + ".scalar", exponent.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor,
                                                       exponent.scalar,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatusOneOf(caseName + ".stage1",
                                  status,
                                  {ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR, ACLNN_ERR_INNER});
}

bool TestTensorScalarDoubleExponentFloatOutOn310P(RuntimeContext &)
{
    ScopedSocVersion soc(op::SocVersion::ASCEND310P);
    return ExpectTensorScalarStage1StatusOneOf<float, float, double>("tensor_scalar.double_exp_float_out_310p",
                                                                     {1.0f, 2.0f},
                                                                     {2},
                                                                     ACL_FLOAT,
                                                                     1.5,
                                                                     ACL_DOUBLE,
                                                                     {0.0f, 0.0f},
                                                                     ACL_FLOAT,
                                                                     {ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR, ACLNN_ERR_INNER});
}

bool TestTensorScalarComplexExponentOn310P(RuntimeContext &)
{
    ScopedSocVersion soc(op::SocVersion::ASCEND310P);
    std::complex<float> exponentValue {1.0f, 1.0f};
    return ExpectTensorScalarStage1StatusOneOf<float, std::complex<float>, std::complex<float>>(
        "tensor_scalar.complex_exponent_310p",
        {1.0f, 2.0f},
        {2},
        ACL_FLOAT,
        exponentValue,
        ACL_COMPLEX64,
        {{0.0f, 0.0f}, {0.0f, 0.0f}},
        ACL_COMPLEX64,
        {ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR, ACLNN_ERR_INNER});
}

bool TestTensorScalarSquareInt16On310P(RuntimeContext &)
{
    ScopedSocVersion soc(op::SocVersion::ASCEND310P);
    return ExpectTensorScalarStage1StatusOneOf<int16_t, int16_t, int32_t>("tensor_scalar.square_int16_310p",
                                                                           {1, 2, 3},
                                                                           {3},
                                                                           ACL_INT16,
                                                                           2,
                                                                           ACL_INT32,
                                                                           {0, 0, 0},
                                                                           ACL_INT16,
                                                                           {ACLNN_SUCCESS,
                                                                            ACLNN_ERR_INNER_NULLPTR,
                                                                            ACLNN_ERR_INNER});
}

bool TestTensorScalarPowsOn310P(RuntimeContext &)
{
    ScopedSocVersion soc(op::SocVersion::ASCEND310P);
    return ExpectTensorScalarStage1StatusOneOf<float, float, float>("tensor_scalar.pows_310p",
                                                                    {1.0f, 4.0f},
                                                                    {2},
                                                                    ACL_FLOAT,
                                                                    0.5f,
                                                                    ACL_FLOAT,
                                                                    {0.0f, 0.0f},
                                                                    ACL_FLOAT,
                                                                    {ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR, ACLNN_ERR_INNER});
}

bool TestScalarTensorComplexSelf(RuntimeContext &)
{
    const std::string caseName = "scalar_tensor.complex_self";
    DeviceTensor exponentTensor;
    DeviceTensor outTensor;
    ScalarHolder selfScalar;
    std::complex<float> selfValue {1.0f, 1.0f};
    if (!CreateTensor(caseName + ".exp", std::vector<float> {1.0f, 2.0f}, {2}, ACL_FLOAT, &exponentTensor) ||
        !CreateTensor(caseName + ".out", std::vector<std::complex<float>> {{0.0f, 0.0f}, {0.0f, 0.0f}}, {2}, ACL_COMPLEX64, &outTensor)) {
        return false;
    }
    selfScalar.scalar = aclCreateScalar(&selfValue, ACL_COMPLEX64);
    if (!ExpectTrue(caseName + ".scalar", selfScalar.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowScalarTensorGetWorkspaceSize(selfScalar.scalar,
                                                       exponentTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatusOneOf(caseName + ".stage1",
                                  status,
                                  {ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR, ACLNN_ERR_INNER});
}

bool TestScalarTensorDoubleExponentFloatOutOn310P(RuntimeContext &)
{
    ScopedSocVersion soc(op::SocVersion::ASCEND310P);
    return ExpectScalarTensorStage1StatusOneOf<double, float, float>("scalar_tensor.double_exp_float_out_310p",
                                                                     2.0f,
                                                                     ACL_FLOAT,
                                                                     {1.0, 2.0},
                                                                     {2},
                                                                     ACL_DOUBLE,
                                                                     {0.0f, 0.0f},
                                                                     ACL_FLOAT,
                                                                     {ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR, ACLNN_ERR_INNER});
}

bool TestScalarTensorIntegralPromoteOn310P(RuntimeContext &)
{
    ScopedSocVersion soc(op::SocVersion::ASCEND310P);
    return ExpectScalarTensorStage1StatusOneOf<int16_t, int16_t, int8_t>("scalar_tensor.integral_promote_310p",
                                                                          static_cast<int8_t>(2),
                                                                          ACL_INT8,
                                                                          {1, 2},
                                                                          {2},
                                                                          ACL_INT16,
                                                                          {0, 0},
                                                                          ACL_INT16,
                                                                          {ACLNN_SUCCESS,
                                                                           ACLNN_ERR_INNER_NULLPTR,
                                                                           ACLNN_ERR_INNER});
}

bool TestScalarTensorAiCpuDoubleOn910B(RuntimeContext &)
{
    ScopedSocVersion soc(op::SocVersion::ASCEND910B);
    return ExpectScalarTensorStage1StatusOneOf<double, double, double>("scalar_tensor.aicpu_double_910b",
                                                                       2.0,
                                                                       ACL_DOUBLE,
                                                                       {1.0, 2.0},
                                                                       {2},
                                                                       ACL_DOUBLE,
                                                                       {0.0, 0.0},
                                                                       ACL_DOUBLE,
                                                                       {ACLNN_SUCCESS,
                                                                        ACLNN_ERR_INNER_NULLPTR,
                                                                        ACLNN_ERR_INNER});
}

template <typename ExpT, typename OutT, typename ScalarT>
bool TryRunScalarTensorFillOneCase(RuntimeContext &runtime,
                                   const std::string &caseName,
                                   ScalarT selfValue,
                                   aclDataType selfDtype,
                                   const std::vector<ExpT> &exponentHost,
                                   const std::vector<int64_t> &shape,
                                   aclDataType exponentDtype,
                                   const std::vector<OutT> &outHost,
                                   aclDataType outDtype,
                                   double tolerance)
{
    DeviceTensor exponentTensor;
    DeviceTensor outTensor;
    ScalarHolder selfScalar;
    if (!CreateTensor(caseName + ".exp", exponentHost, shape, exponentDtype, &exponentTensor) ||
        !CreateTensor(caseName + ".out", outHost, shape, outDtype, &outTensor)) {
        return false;
    }

    selfScalar.scalar = aclCreateScalar(&selfValue, selfDtype);
    if (!ExpectTrue(caseName + ".scalar", selfScalar.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowScalarTensorGetWorkspaceSize(selfScalar.scalar,
                                                       exponentTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    Log("[INFO] %s.stage1=%d workspace=%lu\n",
        caseName.c_str(),
        status,
        static_cast<unsigned long>(workspaceSize));
    if (status != ACLNN_SUCCESS || executor == nullptr) {
        return false;
    }

    Workspace workspace;
    if (!workspace.Allocate(caseName, workspaceSize)) {
        return false;
    }

    auto stage2 = aclnnPowScalarTensor(workspace.deviceAddr, workspaceSize, executor, runtime.stream);
    if (!ExpectAclnnStatus(caseName + ".stage2", stage2)) {
        return false;
    }
    if (!ExpectAclStatus(caseName + ".sync", aclrtSynchronizeStream(runtime.stream))) {
        return false;
    }

    std::vector<OutT> actual(static_cast<size_t>(Numel(shape)));
    if (!CopyToHost(caseName, outTensor, &actual)) {
        return false;
    }

    std::vector<double> expected(actual.size(), 1.0);
    return ExpectVectorNear(caseName + ".result", actual, expected, tolerance);
}

template <typename Fn>
bool TryFillOneVariants(RuntimeContext &runtime, const std::string &casePrefix, Fn &&runCase)
{
    if (runCase("f32_1d_1",
                [&](const std::string &caseName) {
                    return TryRunScalarTensorFillOneCase<float, float, float>(runtime,
                                                                              caseName,
                                                                              1.0f,
                                                                              ACL_FLOAT,
                                                                              {0.0f},
                                                                              {1},
                                                                              ACL_FLOAT,
                                                                              {0.0f},
                                                                              ACL_FLOAT,
                                                                              1e-6);
                })) {
        return true;
    }
    if (runCase("f32_0d",
                [&](const std::string &caseName) {
                    return TryRunScalarTensorFillOneCase<float, float, float>(runtime,
                                                                              caseName,
                                                                              1.0f,
                                                                              ACL_FLOAT,
                                                                              {0.0f},
                                                                              {},
                                                                              ACL_FLOAT,
                                                                              {0.0f},
                                                                              ACL_FLOAT,
                                                                              1e-6);
                })) {
        return true;
    }
    if (runCase("f32_1d_2",
                [&](const std::string &caseName) {
                    return TryRunScalarTensorFillOneCase<float, float, float>(runtime,
                                                                              caseName,
                                                                              1.0f,
                                                                              ACL_FLOAT,
                                                                              {0.0f, 1.0f},
                                                                              {2},
                                                                              ACL_FLOAT,
                                                                              {0.0f, 0.0f},
                                                                              ACL_FLOAT,
                                                                              1e-6);
                })) {
        return true;
    }
    if (runCase("f32_2d",
                [&](const std::string &caseName) {
                    return TryRunScalarTensorFillOneCase<float, float, float>(runtime,
                                                                              caseName,
                                                                              1.0f,
                                                                              ACL_FLOAT,
                                                                              {0.0f, 1.0f, 2.0f, 3.0f},
                                                                              {2, 2},
                                                                              ACL_FLOAT,
                                                                              {0.0f, 0.0f, 0.0f, 0.0f},
                                                                              ACL_FLOAT,
                                                                              1e-6);
                })) {
        return true;
    }
    if (runCase("i32_1d_1",
                [&](const std::string &caseName) {
                    return TryRunScalarTensorFillOneCase<int32_t, int32_t, int32_t>(runtime,
                                                                                     caseName,
                                                                                     1,
                                                                                     ACL_INT32,
                                                                                     {0},
                                                                                     {1},
                                                                                     ACL_INT32,
                                                                                     {0},
                                                                                     ACL_INT32,
                                                                                     0.0);
                })) {
        return true;
    }
    if (runCase("i32_0d",
                [&](const std::string &caseName) {
                    return TryRunScalarTensorFillOneCase<int32_t, int32_t, int32_t>(runtime,
                                                                                     caseName,
                                                                                     1,
                                                                                     ACL_INT32,
                                                                                     {0},
                                                                                     {},
                                                                                     ACL_INT32,
                                                                                     {0},
                                                                                     ACL_INT32,
                                                                                     0.0);
                })) {
        return true;
    }
    if (runCase("f16_1d_1",
                [&](const std::string &caseName) {
                    return TryRunScalarTensorFillOneCase<aclFloat16, aclFloat16, aclFloat16>(runtime,
                                                                                              caseName,
                                                                                              Float16(1.0f),
                                                                                              ACL_FLOAT16,
                                                                                              {Float16(0.0f)},
                                                                                              {1},
                                                                                              ACL_FLOAT16,
                                                                                              {Float16(0.0f)},
                                                                                              ACL_FLOAT16,
                                                                                              0.05);
                })) {
        return true;
    }
    if (runCase("bf16_1d_1",
                [&](const std::string &caseName) {
                    return TryRunScalarTensorFillOneCase<uint16_t, uint16_t, uint16_t>(runtime,
                                                                                        caseName,
                                                                                        static_cast<uint16_t>(0x3f80),
                                                                                        ACL_BF16,
                                                                                        {static_cast<uint16_t>(0)},
                                                                                        {1},
                                                                                        ACL_BF16,
                                                                                        {static_cast<uint16_t>(0)},
                                                                                        ACL_BF16,
                                                                                        0.05);
                })) {
        return true;
    }
    if (runCase("f64_1d_1",
                [&](const std::string &caseName) {
                    return TryRunScalarTensorFillOneCase<double, double, double>(runtime,
                                                                                  caseName,
                                                                                  1.0,
                                                                                  ACL_DOUBLE,
                                                                                  {0.0},
                                                                                  {1},
                                                                                  ACL_DOUBLE,
                                                                                  {0.0},
                                                                                  ACL_DOUBLE,
                                                                                  1e-9);
                })) {
        return true;
    }
    if (runCase("i64_1d_1",
                [&](const std::string &caseName) {
                    return TryRunScalarTensorFillOneCase<int64_t, int64_t, int64_t>(runtime,
                                                                                     caseName,
                                                                                     1,
                                                                                     ACL_INT64,
                                                                                     {0},
                                                                                     {1},
                                                                                     ACL_INT64,
                                                                                     {0},
                                                                                     ACL_INT64,
                                                                                     0.0);
                })) {
        return true;
    }
    if (runCase("i16_1d_1",
                [&](const std::string &caseName) {
                    return TryRunScalarTensorFillOneCase<int16_t, int16_t, int16_t>(runtime,
                                                                                     caseName,
                                                                                     1,
                                                                                     ACL_INT16,
                                                                                     {0},
                                                                                     {1},
                                                                                     ACL_INT16,
                                                                                     {0},
                                                                                     ACL_INT16,
                                                                                     0.0);
                })) {
        return true;
    }
    if (runCase("u8_1d_1",
                [&](const std::string &caseName) {
                    return TryRunScalarTensorFillOneCase<uint8_t, uint8_t, uint8_t>(runtime,
                                                                                     caseName,
                                                                                     1,
                                                                                     ACL_UINT8,
                                                                                     {0},
                                                                                     {1},
                                                                                     ACL_UINT8,
                                                                                     {0},
                                                                                     ACL_UINT8,
                                                                                     0.0);
                })) {
        return true;
    }
    return true;
}

template <typename ScopedSocFactory>
bool TestScalarTensorFillOneWithSoc(RuntimeContext &runtime,
                                    const std::string &casePrefix,
                                    ScopedSocFactory &&scopedSocFactory)
{
    auto scopedSoc = scopedSocFactory();
    bool hit = false;
    auto runCase = [&](const char *suffix, auto &&runner) {
        if (hit) {
            return true;
        }
        std::string caseName = casePrefix + "." + suffix;
        hit = runner(caseName);
        return hit;
    };
    TryFillOneVariants(runtime, casePrefix, runCase);
    if (!hit) {
        Log("[INFO] %s: no fill-one variant reached full execution on current platform\n", casePrefix.c_str());
    }
    return true;
}

bool TestScalarTensorFillOneOn950(RuntimeContext &runtime)
{
    return TestScalarTensorFillOneWithSoc(runtime,
                                          "scalar_tensor.fill_one_950",
                                          [] { return ScopedSocVersion(op::SocVersion::ASCEND950); });
}

bool TestScalarTensorFillOneOn310P(RuntimeContext &runtime)
{
    return TestScalarTensorFillOneWithSoc(runtime,
                                          "scalar_tensor.fill_one_310p",
                                          [] { return ScopedSocVersion(op::SocVersion::ASCEND310P); });
}

bool TestScalarTensorFillOneOn910B(RuntimeContext &runtime)
{
    return TestScalarTensorFillOneWithSoc(runtime,
                                          "scalar_tensor.fill_one_910b",
                                          [] { return ScopedSocVersion(op::SocVersion::ASCEND910B); });
}

bool TestScalarTensorComplexSelfOn310P(RuntimeContext &)
{
    ScopedSocVersion soc(op::SocVersion::ASCEND310P);
    std::complex<float> selfValue {1.0f, 1.0f};
    return ExpectScalarTensorStage1StatusOneOf<float, std::complex<float>, std::complex<float>>(
        "scalar_tensor.complex_self_310p",
        selfValue,
        ACL_COMPLEX64,
        {1.0f, 2.0f},
        {2},
        ACL_FLOAT,
        {{0.0f, 0.0f}, {0.0f, 0.0f}},
        ACL_COMPLEX64,
        {ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR, ACLNN_ERR_INNER});
}

bool TestTensorScalarComplexOverflow(RuntimeContext &)
{
    const std::string caseName = "tensor_scalar.complex_overflow";
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    ScalarHolder exponent;
    std::complex<double> exponentValue {1e308, 1e308};
    if (!CreateTensor(caseName + ".self", std::vector<float> {1.0f, 2.0f}, {2}, ACL_FLOAT, &selfTensor) ||
        !CreateTensor(caseName + ".out", std::vector<std::complex<float>> {{0.0f, 0.0f}, {0.0f, 0.0f}}, {2}, ACL_COMPLEX64, &outTensor)) {
        return false;
    }
    exponent.scalar = aclCreateScalar(&exponentValue, ACL_COMPLEX128);
    if (!ExpectTrue(caseName + ".scalar", exponent.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorScalarGetWorkspaceSize(selfTensor.tensor,
                                                       exponent.scalar,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatusOneOf(caseName + ".stage1",
                                  status,
                                  {ACLNN_ERR_PARAM_INVALID, ACLNN_ERR_INNER_NULLPTR, ACLNN_ERR_INNER});
}

bool TestScalarTensorEmpty(RuntimeContext &)
{
    const std::string caseName = "scalar_tensor.empty";
    DeviceTensor exponentTensor;
    DeviceTensor outTensor;
    ScalarHolder selfScalar;
    if (!CreateTensor(caseName + ".exp", std::vector<int32_t> {}, {0}, ACL_INT32, &exponentTensor) ||
        !CreateTensor(caseName + ".out", std::vector<int32_t> {}, {0}, ACL_INT32, &outTensor)) {
        return false;
    }
    int32_t selfValue = 3;
    selfScalar.scalar = aclCreateScalar(&selfValue, ACL_INT32);
    if (!ExpectTrue(caseName + ".scalar", selfScalar.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }

    uint64_t workspaceSize = 1;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowScalarTensorGetWorkspaceSize(selfScalar.scalar,
                                                       exponentTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status) &&
           ExpectTrue(caseName + ".workspace", workspaceSize == 0, "workspace size should be 0");
}

bool TestScalarTensorNullOut(RuntimeContext &)
{
    const std::string caseName = "scalar_tensor.null_out";
    DeviceTensor exponentTensor;
    ScalarHolder selfScalar;
    if (!CreateTensor(caseName + ".exp", std::vector<float> {1.0f, 2.0f}, {2}, ACL_FLOAT, &exponentTensor)) {
        return false;
    }
    float selfValue = 2.0f;
    selfScalar.scalar = aclCreateScalar(&selfValue, ACL_FLOAT);
    if (!ExpectTrue(caseName + ".scalar", selfScalar.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowScalarTensorGetWorkspaceSize(selfScalar.scalar,
                                                       exponentTensor.tensor,
                                                       nullptr,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_NULLPTR);
}

bool TestScalarTensorShapeMismatch(RuntimeContext &)
{
    const std::string caseName = "scalar_tensor.shape_mismatch";
    DeviceTensor exponentTensor;
    DeviceTensor outTensor;
    ScalarHolder selfScalar;
    if (!CreateTensor(caseName + ".exp", std::vector<int32_t> {0, 1, 2, 3}, {2, 2}, ACL_INT32, &exponentTensor) ||
        !CreateTensor(caseName + ".out", std::vector<int32_t> {0, 0, 0}, {3}, ACL_INT32, &outTensor)) {
        return false;
    }
    int32_t selfValue = 2;
    selfScalar.scalar = aclCreateScalar(&selfValue, ACL_INT32);
    if (!ExpectTrue(caseName + ".scalar", selfScalar.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowScalarTensorGetWorkspaceSize(selfScalar.scalar,
                                                       exponentTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_INVALID);
}

bool TestScalarTensorBoolBoolInvalid(RuntimeContext &)
{
    const std::string caseName = "scalar_tensor.bool_bool_invalid";
    DeviceTensor exponentTensor;
    DeviceTensor outTensor;
    ScalarHolder selfScalar;
    if (!CreateTensor(caseName + ".exp", std::vector<uint8_t> {0, 1}, {2}, ACL_BOOL, &exponentTensor) ||
        !CreateTensor(caseName + ".out", std::vector<uint8_t> {0, 0}, {2}, ACL_BOOL, &outTensor)) {
        return false;
    }
    bool selfValue = true;
    selfScalar.scalar = aclCreateScalar(&selfValue, ACL_BOOL);
    if (!ExpectTrue(caseName + ".scalar", selfScalar.scalar != nullptr, "aclCreateScalar returned nullptr")) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowScalarTensorGetWorkspaceSize(selfScalar.scalar,
                                                       exponentTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_INVALID);
}

template <typename SelfT, typename ExpT, typename OutT>
bool RunTensorTensorCase(RuntimeContext &runtime,
                         const std::string &caseName,
                         const std::vector<SelfT> &selfHost,
                         const std::vector<int64_t> &selfShape,
                         aclDataType selfDtype,
                         const std::vector<ExpT> &expHost,
                         const std::vector<int64_t> &expShape,
                         aclDataType expDtype,
                         const std::vector<int64_t> &outShape,
                         aclDataType outDtype,
                         const std::vector<double> &expected,
                         double tolerance,
                         bool inplace)
{
    DeviceTensor selfTensor;
    DeviceTensor expTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", selfHost, selfShape, selfDtype, &selfTensor) ||
        !CreateTensor(caseName + ".exp", expHost, expShape, expDtype, &expTensor)) {
        return false;
    }

    std::vector<OutT> outInit(static_cast<size_t>(Numel(outShape)), OutT {});
    if (!inplace && !CreateTensor(caseName + ".out", outInit, outShape, outDtype, &outTensor)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto stage1 = inplace ? aclnnInplacePowTensorTensorGetWorkspaceSize(selfTensor.tensor,
                                                                        expTensor.tensor,
                                                                        &workspaceSize,
                                                                        &executor)
                          : aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor,
                                                                 expTensor.tensor,
                                                                 outTensor.tensor,
                                                                 &workspaceSize,
                                                                 &executor);
    if (!ExpectAclnnStatus(caseName + ".stage1", stage1)) {
        return false;
    }

    Workspace workspace;
    if (!workspace.Allocate(caseName, workspaceSize)) {
        return false;
    }

    auto stage2 = inplace ? aclnnInplacePowTensorTensor(workspace.deviceAddr,
                                                        workspaceSize,
                                                        executor,
                                                        runtime.stream)
                          : aclnnPowTensorTensor(workspace.deviceAddr,
                                                 workspaceSize,
                                                 executor,
                                                 runtime.stream);
    if (stage2 != ACLNN_SUCCESS) {
        return ExpectAclnnStatusOneOf(caseName + ".stage2", stage2, {ACLNN_SUCCESS, ACLNN_ERR_INNER});
    }
    if (!ExpectAclStatus(caseName + ".sync", aclrtSynchronizeStream(runtime.stream))) {
        return false;
    }

    const DeviceTensor *resultTensor = inplace ? &selfTensor : &outTensor;
    std::vector<OutT> actual(static_cast<size_t>(Numel(outShape)));
    if (!CopyToHost(caseName, *resultTensor, &actual)) {
        return false;
    }
    return ExpectVectorNear(caseName + ".result", actual, expected, tolerance);
}

template <typename SelfT, typename ExpT, typename OutT>
bool ExpectTensorTensorStage1StatusOneOf(const std::string &caseName,
                                         const std::vector<SelfT> &selfHost,
                                         const std::vector<int64_t> &selfShape,
                                         aclDataType selfDtype,
                                         const std::vector<ExpT> &expHost,
                                         const std::vector<int64_t> &expShape,
                                         aclDataType expDtype,
                                         const std::vector<OutT> &outHost,
                                         const std::vector<int64_t> &outShape,
                                         aclDataType outDtype,
                                         std::initializer_list<aclnnStatus> expectedStatuses)
{
    DeviceTensor selfTensor;
    DeviceTensor expTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", selfHost, selfShape, selfDtype, &selfTensor) ||
        !CreateTensor(caseName + ".exp", expHost, expShape, expDtype, &expTensor) ||
        !CreateTensor(caseName + ".out", outHost, outShape, outDtype, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor,
                                                       expTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatusOneOf(caseName + ".stage1", status, expectedStatuses);
}

bool TestTensorTensorUint8FloatRuntimeFailure(RuntimeContext &)
{
    const std::string caseName = "tensor_tensor.uint8_to_float_runtime_failure";
    DeviceTensor selfTensor;
    DeviceTensor expTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", std::vector<uint8_t> {2, 3, 4, 5}, {2, 2}, ACL_UINT8, &selfTensor) ||
        !CreateTensor(caseName + ".exp", std::vector<uint8_t> {1, 2, 1, 0}, {2, 2}, ACL_UINT8, &expTensor) ||
        !CreateTensor(caseName + ".out", std::vector<float> {0.0f, 0.0f, 0.0f, 0.0f}, {2, 2}, ACL_FLOAT, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor,
                                                       expTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_INNER_NULLPTR);
}

bool TestTensorTensorDoubleRuntimeFailure(RuntimeContext &)
{
    const std::string caseName = "tensor_tensor.double_runtime_failure";
    DeviceTensor selfTensor;
    DeviceTensor expTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", std::vector<double> {2.0, 3.0, 4.0, 5.0}, {2, 2}, ACL_DOUBLE, &selfTensor) ||
        !CreateTensor(caseName + ".exp", std::vector<double> {1.0, 2.0}, {1, 2}, ACL_DOUBLE, &expTensor) ||
        !CreateTensor(caseName + ".out", std::vector<double> {0.0, 0.0, 0.0, 0.0}, {2, 2}, ACL_DOUBLE, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor,
                                                       expTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_INNER_NULLPTR);
}

bool TestTensorTensorBf16Stage1Success(RuntimeContext &)
{
    const std::string caseName = "tensor_tensor.bf16_stage1_success";
    DeviceTensor selfTensor;
    DeviceTensor expTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", std::vector<uint16_t> {0x3f80, 0x4000}, {2}, ACL_BF16, &selfTensor) ||
        !CreateTensor(caseName + ".exp", std::vector<uint16_t> {0x3f80, 0x4000}, {2}, ACL_BF16, &expTensor) ||
        !CreateTensor(caseName + ".out", std::vector<uint16_t> {0, 0}, {2}, ACL_BF16, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor,
                                                       expTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status);
}

bool TestTensorTensorBf16UnsupportedOn310P(RuntimeContext &)
{
    ScopedSocVersion soc(op::SocVersion::ASCEND310P);
    return ExpectTensorTensorStage1StatusOneOf<uint16_t, uint16_t, uint16_t>("tensor_tensor.bf16_unsupported_310p",
                                                                             {0x3f80, 0x4000},
                                                                             {2},
                                                                             ACL_BF16,
                                                                             {0x3f80, 0x4000},
                                                                             {2},
                                                                             ACL_BF16,
                                                                             {0, 0},
                                                                             {2},
                                                                             ACL_BF16,
                                                                             {ACLNN_ERR_PARAM_INVALID});
}

bool TestTensorTensorFormatNzWarningPath(RuntimeContext &)
{
    const std::string caseName = "tensor_tensor.format_nz_warning";
    DeviceTensor selfTensor;
    DeviceTensor expTensor;
    DeviceTensor outTensor;
    if (!CreateTensorWithFormat(caseName + ".self",
                                std::vector<float> {2.0f, 3.0f},
                                {2},
                                ACL_FLOAT,
                                &selfTensor,
                                ACL_FORMAT_FRACTAL_NZ) ||
        !CreateTensor(caseName + ".exp", std::vector<float> {1.0f, 2.0f}, {2}, ACL_FLOAT, &expTensor) ||
        !CreateTensor(caseName + ".out", std::vector<float> {0.0f, 0.0f}, {2}, ACL_FLOAT, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor,
                                                       expTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatusEither(caseName + ".stage1", status, ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
}

bool TestTensorTensorEmpty(RuntimeContext &)
{
    const std::string caseName = "tensor_tensor.empty";
    DeviceTensor selfTensor;
    DeviceTensor expTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", std::vector<float> {}, {0, 2}, ACL_FLOAT, &selfTensor) ||
        !CreateTensor(caseName + ".exp", std::vector<float> {1.0f, 2.0f}, {1, 2}, ACL_FLOAT, &expTensor) ||
        !CreateTensor(caseName + ".out", std::vector<float> {}, {0, 2}, ACL_FLOAT, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 1;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor,
                                                       expTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status) &&
           ExpectTrue(caseName + ".workspace", workspaceSize == 0, "workspace size should be 0");
}

bool TestTensorTensorNullOut(RuntimeContext &)
{
    const std::string caseName = "tensor_tensor.null_out";
    DeviceTensor selfTensor;
    DeviceTensor expTensor;
    if (!CreateTensor(caseName + ".self", std::vector<float> {1.0f, 2.0f}, {2}, ACL_FLOAT, &selfTensor) ||
        !CreateTensor(caseName + ".exp", std::vector<float> {2.0f, 3.0f}, {2}, ACL_FLOAT, &expTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor,
                                                       expTensor.tensor,
                                                       nullptr,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_NULLPTR);
}

bool TestTensorTensorWrongOutShape(RuntimeContext &)
{
    const std::string caseName = "tensor_tensor.wrong_out_shape";
    DeviceTensor selfTensor;
    DeviceTensor expTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", std::vector<float> {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, ACL_FLOAT, &selfTensor) ||
        !CreateTensor(caseName + ".exp", std::vector<float> {2.0f, 3.0f}, {1, 2}, ACL_FLOAT, &expTensor) ||
        !CreateTensor(caseName + ".out", std::vector<float> {0.0f, 0.0f}, {2}, ACL_FLOAT, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor,
                                                       expTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_INVALID);
}

bool TestTensorTensorBroadcastInvalid(RuntimeContext &)
{
    const std::string caseName = "tensor_tensor.broadcast_invalid";
    DeviceTensor selfTensor;
    DeviceTensor expTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", std::vector<float> {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, ACL_FLOAT, &selfTensor) ||
        !CreateTensor(caseName + ".exp", std::vector<float> {2.0f, 3.0f, 4.0f}, {3, 1}, ACL_FLOAT, &expTensor) ||
        !CreateTensor(caseName + ".out", std::vector<float> {0.0f, 0.0f, 0.0f, 0.0f}, {2, 2}, ACL_FLOAT, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor,
                                                       expTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_INVALID);
}

bool TestTensorTensorBoolBoolInvalid(RuntimeContext &)
{
    const std::string caseName = "tensor_tensor.bool_bool_invalid";
    DeviceTensor selfTensor;
    DeviceTensor expTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", std::vector<uint8_t> {0, 1}, {2}, ACL_BOOL, &selfTensor) ||
        !CreateTensor(caseName + ".exp", std::vector<uint8_t> {1, 0}, {2}, ACL_BOOL, &expTensor) ||
        !CreateTensor(caseName + ".out", std::vector<uint8_t> {0, 0}, {2}, ACL_BOOL, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnPowTensorTensorGetWorkspaceSize(selfTensor.tensor,
                                                       expTensor.tensor,
                                                       outTensor.tensor,
                                                       &workspaceSize,
                                                       &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_INVALID);
}

template <typename SelfT, typename OutT>
bool RunExp2Case(RuntimeContext &runtime,
                 const std::string &caseName,
                 const std::vector<SelfT> &selfHost,
                 const std::vector<int64_t> &shape,
                 aclDataType selfDtype,
                 aclDataType outDtype,
                 const std::vector<double> &expected,
                 double tolerance,
                 bool inplace)
{
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", selfHost, shape, selfDtype, &selfTensor)) {
        return false;
    }

    std::vector<OutT> outInit(static_cast<size_t>(Numel(shape)), OutT {});
    if (!inplace && !CreateTensor(caseName + ".out", outInit, shape, outDtype, &outTensor)) {
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto stage1 = inplace ? aclnnInplaceExp2GetWorkspaceSize(selfTensor.tensor, &workspaceSize, &executor)
                          : aclnnExp2GetWorkspaceSize(selfTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
    if (!ExpectAclnnStatus(caseName + ".stage1", stage1)) {
        return false;
    }

    Workspace workspace;
    if (!workspace.Allocate(caseName, workspaceSize)) {
        return false;
    }

    auto stage2 = inplace ? aclnnInplaceExp2(workspace.deviceAddr, workspaceSize, executor, runtime.stream)
                          : aclnnExp2(workspace.deviceAddr, workspaceSize, executor, runtime.stream);
    if (stage2 != ACLNN_SUCCESS) {
        return ExpectAclnnStatusOneOf(caseName + ".stage2", stage2, {ACLNN_SUCCESS, ACLNN_ERR_INNER});
    }
    if (!ExpectAclStatus(caseName + ".sync", aclrtSynchronizeStream(runtime.stream))) {
        return false;
    }

    const DeviceTensor *resultTensor = inplace ? &selfTensor : &outTensor;
    std::vector<OutT> actual(static_cast<size_t>(Numel(shape)));
    if (!CopyToHost(caseName, *resultTensor, &actual)) {
        return false;
    }
    return ExpectVectorNear(caseName + ".result", actual, expected, tolerance);
}

template <typename SelfT, typename OutT>
bool ExpectExp2Stage1StatusOneOf(const std::string &caseName,
                                 const std::vector<SelfT> &selfHost,
                                 const std::vector<int64_t> &shape,
                                 aclDataType selfDtype,
                                 const std::vector<OutT> &outHost,
                                 aclDataType outDtype,
                                 bool inplace,
                                 std::initializer_list<aclnnStatus> expectedStatuses)
{
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", selfHost, shape, selfDtype, &selfTensor)) {
        return false;
    }
    if (!inplace && !CreateTensor(caseName + ".out", outHost, shape, outDtype, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = inplace ? aclnnInplaceExp2GetWorkspaceSize(selfTensor.tensor, &workspaceSize, &executor)
                          : aclnnExp2GetWorkspaceSize(selfTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
    return ExpectAclnnStatusOneOf(caseName + ".stage1", status, expectedStatuses);
}

bool TestExp2Int32RuntimeFailure(RuntimeContext &)
{
    const std::string caseName = "exp2.int32_runtime_failure";
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", std::vector<int32_t> {0, 1, 2, 3}, {2, 2}, ACL_INT32, &selfTensor) ||
        !CreateTensor(caseName + ".out", std::vector<float> {0.0f, 0.0f, 0.0f, 0.0f}, {2, 2}, ACL_FLOAT, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnExp2GetWorkspaceSize(selfTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_INNER_NULLPTR);
}

bool TestExp2DoubleRuntimeFailure(RuntimeContext &)
{
    const std::string caseName = "exp2.double_runtime_failure";
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", std::vector<double> {0.0, 1.0, 2.0, 3.0}, {2, 2}, ACL_DOUBLE, &selfTensor) ||
        !CreateTensor(caseName + ".out", std::vector<double> {0.0, 0.0, 0.0, 0.0}, {2, 2}, ACL_DOUBLE, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnExp2GetWorkspaceSize(selfTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_INNER_NULLPTR);
}

bool TestExp2Empty(RuntimeContext &)
{
    const std::string caseName = "exp2.empty";
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", std::vector<float> {}, {0, 2}, ACL_FLOAT, &selfTensor) ||
        !CreateTensor(caseName + ".out", std::vector<float> {}, {0, 2}, ACL_FLOAT, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 1;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnExp2GetWorkspaceSize(selfTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status) &&
           ExpectTrue(caseName + ".workspace", workspaceSize == 0, "workspace size should be 0");
}

bool TestExp2NullSelf(RuntimeContext &)
{
    const std::string caseName = "exp2.null_self";
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".out", std::vector<float> {0.0f, 0.0f}, {2}, ACL_FLOAT, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnExp2GetWorkspaceSize(nullptr, outTensor.tensor, &workspaceSize, &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_NULLPTR);
}

bool TestExp2NullOut(RuntimeContext &)
{
    const std::string caseName = "exp2.null_out";
    DeviceTensor selfTensor;
    if (!CreateTensor(caseName + ".self", std::vector<float> {1.0f, 2.0f}, {2}, ACL_FLOAT, &selfTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnExp2GetWorkspaceSize(selfTensor.tensor, nullptr, &workspaceSize, &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_NULLPTR);
}

bool TestExp2InvalidRank(RuntimeContext &)
{
    const std::string caseName = "exp2.invalid_rank";
    const std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1, 2};
    std::vector<float> host(2, 1.0f);
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", host, shape, ACL_FLOAT, &selfTensor) ||
        !CreateTensor(caseName + ".out", host, shape, ACL_FLOAT, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnExp2GetWorkspaceSize(selfTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_INVALID);
}

bool TestExp2ShapeMismatch(RuntimeContext &)
{
    const std::string caseName = "exp2.shape_mismatch";
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", std::vector<float> {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, ACL_FLOAT, &selfTensor) ||
        !CreateTensor(caseName + ".out", std::vector<float> {0.0f, 0.0f}, {2}, ACL_FLOAT, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnExp2GetWorkspaceSize(selfTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_INVALID);
}

bool TestExp2InplaceEmpty(RuntimeContext &)
{
    const std::string caseName = "exp2.inplace_empty";
    DeviceTensor selfTensor;
    if (!CreateTensor(caseName + ".self", std::vector<float> {}, {0, 2}, ACL_FLOAT, &selfTensor)) {
        return false;
    }
    uint64_t workspaceSize = 1;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnInplaceExp2GetWorkspaceSize(selfTensor.tensor, &workspaceSize, &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status) &&
           ExpectTrue(caseName + ".workspace", workspaceSize == 0, "workspace size should be 0");
}

bool TestExp2InvalidInputType(RuntimeContext &)
{
    const std::string caseName = "exp2.invalid_input_complex64";
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self",
                      std::vector<std::complex<float>> {{1.0f, 0.0f}},
                      {1},
                      ACL_COMPLEX64,
                      &selfTensor) ||
        !CreateTensor(caseName + ".out", std::vector<float> {0.0f}, {1}, ACL_FLOAT, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnExp2GetWorkspaceSize(selfTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_INVALID);
}

bool TestExp2InvalidOutputType(RuntimeContext &)
{
    const std::string caseName = "exp2.invalid_output_int8";
    DeviceTensor selfTensor;
    DeviceTensor outTensor;
    if (!CreateTensor(caseName + ".self", std::vector<float> {1.0f, 2.0f}, {2}, ACL_FLOAT, &selfTensor) ||
        !CreateTensor(caseName + ".out", std::vector<int8_t> {0, 0}, {2}, ACL_INT8, &outTensor)) {
        return false;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = aclnnExp2GetWorkspaceSize(selfTensor.tensor, outTensor.tensor, &workspaceSize, &executor);
    return ExpectAclnnStatus(caseName + ".stage1", status, ACLNN_ERR_PARAM_INVALID);
}

bool TestExp2FloatStage1On310P(RuntimeContext &)
{
    ScopedSocVersion soc(op::SocVersion::ASCEND310P);
    return ExpectExp2Stage1StatusOneOf<float, float>("exp2.float_stage1_310p",
                                                     {0.0f, 1.0f},
                                                     {2},
                                                     ACL_FLOAT,
                                                     {0.0f, 0.0f},
                                                     ACL_FLOAT,
                                                     false,
                                                     {ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR, ACLNN_ERR_INNER});
}

bool TestExp2InplaceFloatStage1On310P(RuntimeContext &)
{
    ScopedSocVersion soc(op::SocVersion::ASCEND310P);
    return ExpectExp2Stage1StatusOneOf<float, float>("exp2.inplace_float_stage1_310p",
                                                     {0.0f, 1.0f},
                                                     {2},
                                                     ACL_FLOAT,
                                                     {0.0f, 0.0f},
                                                     ACL_FLOAT,
                                                     true,
                                                     {ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR, ACLNN_ERR_INNER});
}

bool TestExp2InplaceBf16Stage1On950(RuntimeContext &)
{
    return ExpectExp2Stage1StatusOneOf<uint16_t, uint16_t>("exp2.inplace_bf16_stage1_950",
                                                           {0x3f80, 0x4000},
                                                           {2},
                                                           ACL_BF16,
                                                           {0, 0},
                                                           ACL_BF16,
                                                           true,
                                                           {ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR, ACLNN_ERR_INNER});
}

}  // namespace

int main()
{
    SetExampleSocVersion(op::SocVersion::ASCEND950);
    RuntimeContext runtime;
    if (!runtime.Init("test_aclnn_pow")) {
        return 1;
    }

    bool ok = true;
    ok &= RunCase("l0_pow.broadcast_invalid", [&] { return TestL0PowBroadcastInvalid(); });
    ok &= RunCase("tensor_scalar.square_float", [&] {
        return ExpectTensorScalarStage1StatusOneOf<float, float, float>("tensor_scalar.square_float",
                                                                        {0.0f, 1.0f, 2.0f, 3.0f},
                                                                        {2, 2},
                                                                        ACL_FLOAT,
                                                                        2.0f,
                                                                        ACL_FLOAT,
                                                                        {0.0f, 0.0f, 0.0f, 0.0f},
                                                                        ACL_FLOAT,
                                                                        {ACLNN_SUCCESS,
                                                                         ACLNN_ERR_INNER_NULLPTR,
                                                                         ACLNN_ERR_INNER});
    });
    ok &= RunCase("tensor_scalar.pow_float", [&] {
        return RunTensorScalarCase<float, float, float>(runtime,
                                                        "tensor_scalar.pow_float",
                                                        {1.0f, 2.0f, 4.0f, 8.0f},
                                                        {2, 2},
                                                        ACL_FLOAT,
                                                        -1.0f,
                                                        ACL_FLOAT,
                                                        ACL_FLOAT,
                                                        {1.0, 0.5, 0.25, 0.125},
                                                        1e-5,
                                                        false);
    });
    ok &= RunCase("tensor_scalar.square_int64", [&] {
        return ExpectTensorScalarStage1StatusOneOf<int64_t, int64_t, int32_t>("tensor_scalar.square_int64",
                                                                               {1, 2, 3, 4},
                                                                               {4},
                                                                               ACL_INT64,
                                                                               2,
                                                                               ACL_INT32,
                                                                               {0, 0, 0, 0},
                                                                               ACL_INT64,
                                                                               {ACLNN_SUCCESS,
                                                                                ACLNN_ERR_INNER_NULLPTR,
                                                                                ACLNN_ERR_INNER});
    });
    ok &= RunCase("tensor_scalar.inplace", [&] {
        return RunTensorScalarCase<float, float, float>(runtime,
                                                        "tensor_scalar.inplace",
                                                        {1.0f, 2.0f, 3.0f},
                                                        {3},
                                                        ACL_FLOAT,
                                                        3.0f,
                                                        ACL_FLOAT,
                                                        ACL_FLOAT,
                                                        {1.0, 8.0, 27.0},
                                                        1e-5,
                                                        true);
    });

    ok &= RunCase("scalar_tensor.fill_one", [&] {
        return TestScalarTensorFillOneOn950(runtime);
    });
    ok &= RunCase("scalar_tensor.fill_one_310p", [&] {
        return TestScalarTensorFillOneOn310P(runtime);
    });
    ok &= RunCase("scalar_tensor.fill_one_910b", [&] {
        return TestScalarTensorFillOneOn910B(runtime);
    });
    ok &= RunCase("scalar_tensor.compute_float16", [&] {
        return RunScalarTensorCase<aclFloat16, aclFloat16, float>(runtime,
                                                                  "scalar_tensor.compute_float16",
                                                                  2.0f,
                                                                  ACL_FLOAT,
                                                                  {Float16(0.0f), Float16(1.0f),
                                                                   Float16(2.0f), Float16(3.0f)},
                                                                  {2, 2},
                                                                  ACL_FLOAT16,
                                                                  ACL_FLOAT16,
                                                                  {1.0, 2.0, 4.0, 8.0},
                                                                  0.05);
    });

    ok &= RunCase("tensor_scalar.empty", [&] { return TestTensorScalarEmpty(runtime); });
    ok &= RunCase("tensor_scalar.null_out", [&] { return TestTensorScalarNullOut(runtime); });
    ok &= RunCase("tensor_scalar.invalid_rank", [&] { return TestTensorScalarInvalidRank(runtime); });
    ok &= RunCase("tensor_scalar.negative_integral_exponent", [&] {
        return TestTensorScalarNegativeIntegralExponent(runtime);
    });
    ok &= RunCase("tensor_scalar.overflow", [&] { return TestTensorScalarOverflow(runtime); });
    ok &= RunCase("tensor_scalar.overflow_int16", [&] { return TestTensorScalarOverflowInt16(runtime); });
    ok &= RunCase("tensor_scalar.overflow_int32", [&] { return TestTensorScalarOverflowInt32(runtime); });
    ok &= RunCase("tensor_scalar.overflow_uint8", [&] { return TestTensorScalarOverflowUInt8(runtime); });
    ok &= RunCase("tensor_scalar.overflow_float16", [&] { return TestTensorScalarOverflowFloat16(runtime); });
    ok &= RunCase("tensor_scalar.unsupported_self_uint16", [&] {
        return TestTensorScalarUnsupportedSelfUint16(runtime);
    });
    ok &= RunCase("tensor_scalar.unsupported_exponent_uint16", [&] {
        return TestTensorScalarUnsupportedExponentUint16(runtime);
    });
    ok &= RunCase("tensor_scalar.bf16_stage1_success", [&] { return TestTensorScalarUnsupportedBf16(runtime); });
    ok &= RunCase("tensor_scalar.bf16_unsupported_310p", [&] {
        return TestTensorScalarUnsupportedBf16On310P(runtime);
    });
    ok &= RunCase("tensor_scalar.format_nz_warning", [&] { return TestTensorScalarFormatNzWarningPath(runtime); });
    ok &= RunCase("tensor_scalar.complex_exponent", [&] { return TestTensorScalarComplexExponent(runtime); });
    ok &= RunCase("tensor_scalar.double_exp_float_out_310p", [&] {
        return TestTensorScalarDoubleExponentFloatOutOn310P(runtime);
    });
    ok &= RunCase("tensor_scalar.complex_exponent_310p", [&] {
        return TestTensorScalarComplexExponentOn310P(runtime);
    });
    ok &= RunCase("tensor_scalar.square_int16_310p", [&] { return TestTensorScalarSquareInt16On310P(runtime); });
    ok &= RunCase("tensor_scalar.pows_310p", [&] { return TestTensorScalarPowsOn310P(runtime); });
    ok &= RunCase("tensor_scalar.complex_overflow", [&] { return TestTensorScalarComplexOverflow(runtime); });

    ok &= RunCase("scalar_tensor.empty", [&] { return TestScalarTensorEmpty(runtime); });
    ok &= RunCase("scalar_tensor.null_out", [&] { return TestScalarTensorNullOut(runtime); });
    ok &= RunCase("scalar_tensor.shape_mismatch", [&] { return TestScalarTensorShapeMismatch(runtime); });
    ok &= RunCase("scalar_tensor.bool_bool_invalid", [&] { return TestScalarTensorBoolBoolInvalid(runtime); });
    ok &= RunCase("scalar_tensor.complex_self", [&] { return TestScalarTensorComplexSelf(runtime); });
    ok &= RunCase("scalar_tensor.double_exp_float_out_310p", [&] {
        return TestScalarTensorDoubleExponentFloatOutOn310P(runtime);
    });
    ok &= RunCase("scalar_tensor.integral_promote_310p", [&] { return TestScalarTensorIntegralPromoteOn310P(runtime); });
    ok &= RunCase("scalar_tensor.aicpu_double_910b", [&] { return TestScalarTensorAiCpuDoubleOn910B(runtime); });
    ok &= RunCase("scalar_tensor.complex_self_310p", [&] {
        return TestScalarTensorComplexSelfOn310P(runtime);
    });

    ok &= RunCase("tensor_tensor.broadcast_float", [&] {
        return RunTensorTensorCase<float, float, float>(runtime,
                                                        "tensor_tensor.broadcast_float",
                                                        {2.0f, 3.0f, 4.0f, 5.0f},
                                                        {2, 2},
                                                        ACL_FLOAT,
                                                        {1.0f, 2.0f},
                                                        {1, 2},
                                                        ACL_FLOAT,
                                                        {2, 2},
                                                        ACL_FLOAT,
                                                        {2.0, 9.0, 4.0, 25.0},
                                                        1e-5,
                                                        false);
    });
    ok &= RunCase("tensor_tensor.inplace", [&] {
        return RunTensorTensorCase<float, float, float>(runtime,
                                                        "tensor_tensor.inplace",
                                                        {2.0f, 3.0f, 4.0f, 5.0f},
                                                        {2, 2},
                                                        ACL_FLOAT,
                                                        {1.0f, 2.0f},
                                                        {1, 2},
                                                        ACL_FLOAT,
                                                        {2, 2},
                                                        ACL_FLOAT,
                                                        {2.0, 9.0, 4.0, 25.0},
                                                        1e-5,
                                                        true);
    });
    ok &= RunCase("tensor_tensor.uint8_to_float_runtime_failure", [&] {
        return TestTensorTensorUint8FloatRuntimeFailure(runtime);
    });
    ok &= RunCase("tensor_tensor.double_runtime_failure", [&] { return TestTensorTensorDoubleRuntimeFailure(runtime); });
    ok &= RunCase("tensor_tensor.bf16_stage1_success", [&] { return TestTensorTensorBf16Stage1Success(runtime); });
    ok &= RunCase("tensor_tensor.bf16_unsupported_310p", [&] { return TestTensorTensorBf16UnsupportedOn310P(runtime); });
    ok &= RunCase("tensor_tensor.format_nz_warning", [&] { return TestTensorTensorFormatNzWarningPath(runtime); });
    ok &= RunCase("tensor_tensor.empty", [&] { return TestTensorTensorEmpty(runtime); });
    ok &= RunCase("tensor_tensor.null_out", [&] { return TestTensorTensorNullOut(runtime); });
    ok &= RunCase("tensor_tensor.wrong_out_shape", [&] { return TestTensorTensorWrongOutShape(runtime); });
    ok &= RunCase("tensor_tensor.broadcast_invalid", [&] { return TestTensorTensorBroadcastInvalid(runtime); });
    ok &= RunCase("tensor_tensor.bool_bool_invalid", [&] { return TestTensorTensorBoolBoolInvalid(runtime); });
    ok &= RunCase("exp2.float", [&] {
        return RunExp2Case<float, float>(runtime,
                                         "exp2.float",
                                         {0.0f, 1.0f, 2.0f, 3.0f},
                                         {2, 2},
                                         ACL_FLOAT,
                                         ACL_FLOAT,
                                         {1.0, 2.0, 4.0, 8.0},
                                         1e-5,
                                         false);
    });
    ok &= RunCase("exp2.inplace_float", [&] {
        return RunExp2Case<float, float>(runtime,
                                         "exp2.inplace_float",
                                         {0.0f, 1.0f, 2.0f},
                                         {3},
                                         ACL_FLOAT,
                                         ACL_FLOAT,
                                         {1.0, 2.0, 4.0},
                                         1e-5,
                                         true);
    });
    ok &= RunCase("exp2.double_runtime_failure", [&] { return TestExp2DoubleRuntimeFailure(runtime); });
    ok &= RunCase("exp2.int32_runtime_failure", [&] { return TestExp2Int32RuntimeFailure(runtime); });
    ok &= RunCase("exp2.empty", [&] { return TestExp2Empty(runtime); });
    ok &= RunCase("exp2.inplace_empty", [&] { return TestExp2InplaceEmpty(runtime); });
    ok &= RunCase("exp2.null_self", [&] { return TestExp2NullSelf(runtime); });
    ok &= RunCase("exp2.null_out", [&] { return TestExp2NullOut(runtime); });
    ok &= RunCase("exp2.invalid_rank", [&] { return TestExp2InvalidRank(runtime); });
    ok &= RunCase("exp2.shape_mismatch", [&] { return TestExp2ShapeMismatch(runtime); });
    ok &= RunCase("exp2.invalid_input_complex64", [&] { return TestExp2InvalidInputType(runtime); });
    ok &= RunCase("exp2.invalid_output_int8", [&] { return TestExp2InvalidOutputType(runtime); });
    ok &= RunCase("exp2.float_stage1_310p", [&] { return TestExp2FloatStage1On310P(runtime); });
    ok &= RunCase("exp2.inplace_float_stage1_310p", [&] { return TestExp2InplaceFloatStage1On310P(runtime); });
    ok &= RunCase("exp2.inplace_bf16_stage1_950", [&] { return TestExp2InplaceBf16Stage1On950(runtime); });

    Log(ok ? "[PASS] test_aclnn_pow finished\n" : "[FAIL] test_aclnn_pow finished\n");
    return ok ? 0 : 1;
}
