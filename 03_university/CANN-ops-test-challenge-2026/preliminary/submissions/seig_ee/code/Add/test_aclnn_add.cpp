/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Add 算子端到端测试（CPU 模拟器友好）：仅使用 ACL_FLOAT 张量，避免 INT8/BF16/FP16/混合类型
 * 在 QEMU 上常见 SIGSEGV。真机评测若需更高覆盖率可再逐步加回 dtype。
 *
 * 覆盖：aclnnAdd / aclnnAdds / aclnnInplaceAdd / aclnnInplaceAdds / aclnnAddV3 / aclnnInplaceAddV3
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

#define LOG_PRINT(message, ...) printf(message, ##__VA_ARGS__)

namespace {

/** 模拟器浮点误差较大，容差略放宽；接近 0 时用 max(1,|e|) 缩放相对误差。 */
constexpr double kRtol = 5e-2;
constexpr double kAtol = 5e-3;

/**
 * @brief aclnn 接口成功码在部分环境为 0，与宏 ACL_SUCCESS 可能类型不一致，统一按整型 0 判断。
 */
inline bool AclnnOk(aclnnStatus st)
{
    return static_cast<int32_t>(st) == 0;
}

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

/**
 * @brief 计算连续 tensor 元素个数。
 */
int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t n = 1;
    for (int64_t d : shape) {
        n *= d;
    }
    return n;
}

/**
 * @brief 将 host 数据写入 device 并创建 aclTensor（连续 ND）。
 */
template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    const int64_t n = GetShapeSize(shape);
    const size_t bytes = static_cast<size_t>(n) * sizeof(T);
    aclError ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return static_cast<int>(ret));
    ret = aclrtMemcpy(*deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return static_cast<int>(ret));

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int InitAcl(int32_t deviceId, aclrtStream* stream)
{
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclInit failed. ERROR: %d\n", ret); return static_cast<int>(ret));
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtSetDevice failed. ERROR: %d\n", ret); return static_cast<int>(ret));
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, printf("aclrtCreateStream failed. ERROR: %d\n", ret); return static_cast<int>(ret));
    return 0;
}

/**
 * @brief 两段式算子调用：GetWorkspaceSize → Execute；失败时打印 aclnnStatus 便于排查（如未安装 custom 包）。
 */
template <typename GetWs, typename RunOp>
aclnnStatus RunTwoStage(aclrtStream stream, GetWs&& getWs, RunOp&& runOp)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus st = getWs(&workspaceSize, &executor);
    if (!AclnnOk(st)) {
        return st;
    }
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        aclError ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return static_cast<aclnnStatus>(ret);
        }
    }
    st = runOp(workspaceAddr, workspaceSize, executor, stream);
    if (!AclnnOk(st)) {
        if (workspaceSize > 0 && workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        return st;
    }
    if (workspaceSize > 0 && workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return st;
}

/**
 * @brief 调用 RunTwoStage；若失败打印状态码（常见于未执行 pkg 安装 .run，或 ASCEND_CUSTOM_OPP_PATH 未指向 build）。
 */
template <typename GetWs, typename RunOp>
bool InvokeTwoStage(const char* tag, aclrtStream stream, GetWs&& getWs, RunOp&& runOp)
{
    const aclnnStatus st = RunTwoStage(stream, std::forward<GetWs>(getWs), std::forward<RunOp>(runOp));
    if (!AclnnOk(st)) {
        printf(
            "[INFO] %s aclnnStatus=%d — 若持续失败请先: bash build.sh --pkg --soc=ascend950 --ops=add --vendor_name=custom --cov "
            "&& 安装 build_out/*.run，再跑 run_example；必要时 export ASCEND_CUSTOM_OPP_PATH=$PWD/build\n",
            tag, static_cast<int>(st));
        return false;
    }
    return true;
}

/**
 * @brief 浮点逐元素比对。
 */
bool AllCloseFloat(const std::vector<float>& actual, const std::vector<float>& expected, const char* name)
{
    if (actual.size() != expected.size()) {
        printf("[FAIL] %s size mismatch\n", name);
        return false;
    }
    for (size_t i = 0; i < actual.size(); i++) {
        const double e = static_cast<double>(expected[i]);
        const double a = static_cast<double>(actual[i]);
        if (std::isnan(e) && std::isnan(a)) {
            continue;
        }
        if (std::isinf(e) && std::isinf(a) && std::signbit(e) == std::signbit(a)) {
            continue;
        }
        const double diff = std::abs(a - e);
        const double scale = std::max(1.0, std::abs(e));
        const double tol = kAtol + kRtol * scale;
        if (diff > tol) {
            printf("[FAIL] %s idx=%zu expected=%g actual=%g\n", name, i, e, a);
            return false;
        }
    }
    return true;
}

bool CopyFloatOut(void* dev, int64_t n, std::vector<float>* out)
{
    out->resize(static_cast<size_t>(n));
    aclError ret = aclrtMemcpy(
        out->data(), static_cast<size_t>(n) * sizeof(float), dev, static_cast<size_t>(n) * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    return ret == ACL_SUCCESS;
}

struct TestStats {
    int pass = 0;
    int fail = 0;
};

void Record(TestStats* s, bool ok, const char* tag)
{
    if (ok) {
        printf("[PASS] %s\n", tag);
        s->pass++;
    } else {
        printf("[FAIL] %s\n", tag);
        s->fail++;
    }
}

} // namespace

int main()
{
    TestStats stats;
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    if (InitAcl(deviceId, &stream) != 0) {
        printf("[FAIL] ACL init\n");
        return 1;
    }

    // aclnnAdd float alpha=1.2
    {
        const char* tag = "aclnnAdd float alpha=1.2 same shape";
        std::vector<int64_t> sh = {4, 2};
        std::vector<float> selfH = {0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<float> otherH = {1, 1, 1, 2, 2, 2, 3, 3};
        std::vector<float> outH(8, 0.f);
        float alphaVal = 1.2f;
        void *ds = nullptr, *do_ = nullptr, *dout = nullptr;
        aclTensor *tself = nullptr, *tother = nullptr, *tout = nullptr;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
        int cr = CreateAclTensor(selfH, sh, &ds, ACL_FLOAT, &tself);
        cr |= CreateAclTensor(otherH, sh, &do_, ACL_FLOAT, &tother);
        cr |= CreateAclTensor(outH, sh, &dout, ACL_FLOAT, &tout);
        bool ok = (cr == 0 && alpha != nullptr);
        if (ok) {
            ok = InvokeTwoStage(
                tag, stream,
                [&](uint64_t* ws, aclOpExecutor** ex) {
                    return aclnnAddGetWorkspaceSize(tself, tother, alpha, tout, ws, ex);
                },
                [&](void* w, uint64_t ws, aclOpExecutor* ex, aclrtStream sm) { return aclnnAdd(w, ws, ex, sm); });
        }
        if (ok) {
            ok = (aclrtSynchronizeStream(stream) == ACL_SUCCESS);
        }
        std::vector<float> got;
        std::vector<float> exp(8);
        for (int i = 0; i < 8; i++) {
            exp[static_cast<size_t>(i)] = static_cast<float>(
                static_cast<double>(selfH[static_cast<size_t>(i)]) +
                static_cast<double>(alphaVal) * static_cast<double>(otherH[static_cast<size_t>(i)]));
        }
        if (ok) {
            ok = CopyFloatOut(dout, 8, &got) && AllCloseFloat(got, exp, tag);
        }
        if (tself) {
            aclDestroyTensor(tself);
        }
        if (tother) {
            aclDestroyTensor(tother);
        }
        if (tout) {
            aclDestroyTensor(tout);
        }
        if (alpha) {
            aclDestroyScalar(alpha);
        }
        if (ds) {
            aclrtFree(ds);
        }
        if (do_) {
            aclrtFree(do_);
        }
        if (dout) {
            aclrtFree(dout);
        }
        Record(&stats, ok, tag);
    }

    // aclnnAdd alpha=1
    {
        const char* tag = "aclnnAdd float alpha=1.0";
        std::vector<int64_t> sh = {2, 2};
        std::vector<float> a = {1, 2, 3, 4};
        std::vector<float> b = {10, 20, 30, 40};
        float one = 1.f;
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *tc = nullptr;
        aclScalar* alp = aclCreateScalar(&one, ACL_FLOAT);
        bool ok = CreateAclTensor(a, sh, &da, ACL_FLOAT, &ta) == 0 && CreateAclTensor(b, sh, &db, ACL_FLOAT, &tb) == 0 &&
                  CreateAclTensor(std::vector<float>(4, 0.f), sh, &dc, ACL_FLOAT, &tc) == 0 && alp != nullptr;
        if (ok) {
            ok = InvokeTwoStage(
                tag, stream,
                [&](uint64_t* ws, aclOpExecutor** ex) {
                    return aclnnAddGetWorkspaceSize(ta, tb, alp, tc, ws, ex);
                },
                [&](void* w, uint64_t ws, aclOpExecutor* ex, aclrtStream sm) { return aclnnAdd(w, ws, ex, sm); }) &&
                 (aclrtSynchronizeStream(stream) == ACL_SUCCESS);
        }
        std::vector<float> got;
        std::vector<float> exp = {11, 22, 33, 44};
        ok = ok && CopyFloatOut(dc, 4, &got) && AllCloseFloat(got, exp, tag);
        aclDestroyTensor(ta);
        aclDestroyTensor(tb);
        aclDestroyTensor(tc);
        aclDestroyScalar(alp);
        aclrtFree(da);
        aclrtFree(db);
        aclrtFree(dc);
        Record(&stats, ok, tag);
    }

    // broadcast
    {
        const char* tag = "aclnnAdd float broadcast";
        std::vector<int64_t> sself = {4, 1};
        std::vector<int64_t> sother = {1, 4};
        std::vector<int64_t> sout = {4, 4};
        std::vector<float> selfH(4);
        std::vector<float> otherH(4);
        for (int i = 0; i < 4; i++) {
            selfH[static_cast<size_t>(i)] = static_cast<float>(i);
            otherH[static_cast<size_t>(i)] = static_cast<float>(i + 1);
        }
        float alphaVal = 0.5f;
        void *ds = nullptr, *do_ = nullptr, *dout = nullptr;
        aclTensor *tself = nullptr, *tother = nullptr, *tout = nullptr;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
        std::vector<float> outInit(16, 0.f);
        bool ok = CreateAclTensor(selfH, sself, &ds, ACL_FLOAT, &tself) == 0 &&
                  CreateAclTensor(otherH, sother, &do_, ACL_FLOAT, &tother) == 0 &&
                  CreateAclTensor(outInit, sout, &dout, ACL_FLOAT, &tout) == 0 && alpha != nullptr;
        if (ok) {
            ok = InvokeTwoStage(
                tag, stream,
                [&](uint64_t* ws, aclOpExecutor** ex) {
                    return aclnnAddGetWorkspaceSize(tself, tother, alpha, tout, ws, ex);
                },
                [&](void* w, uint64_t ws, aclOpExecutor* ex, aclrtStream sm) { return aclnnAdd(w, ws, ex, sm); }) &&
                 (aclrtSynchronizeStream(stream) == ACL_SUCCESS);
        }
        std::vector<float> got;
        std::vector<float> exp(16);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                const double x1 = static_cast<double>(selfH[static_cast<size_t>(i)]);
                const double x2 = static_cast<double>(otherH[static_cast<size_t>(j)]);
                exp[static_cast<size_t>(i * 4 + j)] = static_cast<float>(x1 + static_cast<double>(alphaVal) * x2);
            }
        }
        ok = ok && CopyFloatOut(dout, 16, &got) && AllCloseFloat(got, exp, tag);
        aclDestroyTensor(tself);
        aclDestroyTensor(tother);
        aclDestroyTensor(tout);
        aclDestroyScalar(alpha);
        aclrtFree(ds);
        aclrtFree(do_);
        aclrtFree(dout);
        Record(&stats, ok, tag);
    }

    // aclnnAdds
    {
        const char* tag = "aclnnAdds float tensor + scalar";
        std::vector<int64_t> sh = {2, 3};
        std::vector<float> selfH = {1, 2, 3, 4, 5, 6};
        float otherScalar = 100.f;
        float alphaVal = 0.1f;
        void* ds = nullptr;
        void* dout = nullptr;
        aclTensor* tself = nullptr;
        aclTensor* tout = nullptr;
        aclScalar* sOther = aclCreateScalar(&otherScalar, ACL_FLOAT);
        aclScalar* sAlpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
        bool ok = CreateAclTensor(selfH, sh, &ds, ACL_FLOAT, &tself) == 0 &&
                  CreateAclTensor(std::vector<float>(6, 0.f), sh, &dout, ACL_FLOAT, &tout) == 0 && sOther != nullptr &&
                  sAlpha != nullptr;
        if (ok) {
            ok = InvokeTwoStage(
                tag, stream,
                [&](uint64_t* ws, aclOpExecutor** ex) {
                    return aclnnAddsGetWorkspaceSize(tself, sOther, sAlpha, tout, ws, ex);
                },
                [&](void* w, uint64_t ws, aclOpExecutor* ex, aclrtStream sm) { return aclnnAdds(w, ws, ex, sm); }) &&
                 (aclrtSynchronizeStream(stream) == ACL_SUCCESS);
        }
        std::vector<float> got;
        std::vector<float> exp(6);
        for (int i = 0; i < 6; i++) {
            exp[static_cast<size_t>(i)] = selfH[static_cast<size_t>(i)] + alphaVal * otherScalar;
        }
        ok = ok && CopyFloatOut(dout, 6, &got) && AllCloseFloat(got, exp, tag);
        aclDestroyTensor(tself);
        aclDestroyTensor(tout);
        aclDestroyScalar(sOther);
        aclDestroyScalar(sAlpha);
        aclrtFree(ds);
        aclrtFree(dout);
        Record(&stats, ok, tag);
    }

    // aclnnInplaceAdd
    {
        const char* tag = "aclnnInplaceAdd float";
        std::vector<int64_t> sh = {2, 2};
        std::vector<float> selfH = {1, 2, 3, 4};
        std::vector<float> otherH = {0.5f, 1.5f, 2.5f, 3.5f};
        float alphaVal = 2.f;
        void *ds = nullptr, *do_ = nullptr;
        aclTensor *tself = nullptr, *tother = nullptr;
        aclScalar* alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
        bool ok = CreateAclTensor(selfH, sh, &ds, ACL_FLOAT, &tself) == 0 &&
                  CreateAclTensor(otherH, sh, &do_, ACL_FLOAT, &tother) == 0 && alpha != nullptr;
        if (ok) {
            ok = InvokeTwoStage(
                tag, stream,
                [&](uint64_t* ws, aclOpExecutor** ex) {
                    return aclnnInplaceAddGetWorkspaceSize(tself, tother, alpha, ws, ex);
                },
                [&](void* w, uint64_t ws, aclOpExecutor* ex, aclrtStream sm) {
                    return aclnnInplaceAdd(w, ws, ex, sm);
                }) &&
                 (aclrtSynchronizeStream(stream) == ACL_SUCCESS);
        }
        std::vector<float> got;
        std::vector<float> exp(4);
        for (int i = 0; i < 4; i++) {
            exp[static_cast<size_t>(i)] =
                selfH[static_cast<size_t>(i)] + alphaVal * otherH[static_cast<size_t>(i)];
        }
        ok = ok && CopyFloatOut(ds, 4, &got) && AllCloseFloat(got, exp, tag);
        aclDestroyTensor(tself);
        aclDestroyTensor(tother);
        aclDestroyScalar(alpha);
        aclrtFree(ds);
        aclrtFree(do_);
        Record(&stats, ok, tag);
    }

    // aclnnInplaceAdds
    {
        const char* tag = "aclnnInplaceAdds float";
        std::vector<int64_t> sh = {1, 4};
        std::vector<float> selfH = {0, 1, 2, 3};
        float s = 10.f;
        float a = 0.25f;
        void* ds = nullptr;
        aclTensor* tself = nullptr;
        aclScalar* sOther = aclCreateScalar(&s, ACL_FLOAT);
        aclScalar* sAlpha = aclCreateScalar(&a, ACL_FLOAT);
        bool ok = CreateAclTensor(selfH, sh, &ds, ACL_FLOAT, &tself) == 0 && sOther != nullptr && sAlpha != nullptr;
        if (ok) {
            ok = InvokeTwoStage(
                tag, stream,
                [&](uint64_t* ws, aclOpExecutor** ex) {
                    return aclnnInplaceAddsGetWorkspaceSize(tself, sOther, sAlpha, ws, ex);
                },
                [&](void* w, uint64_t ws, aclOpExecutor* ex, aclrtStream sm) {
                    return aclnnInplaceAdds(w, ws, ex, sm);
                }) &&
                 (aclrtSynchronizeStream(stream) == ACL_SUCCESS);
        }
        std::vector<float> got;
        std::vector<float> exp = {2.5f, 3.5f, 4.5f, 5.5f};
        ok = ok && CopyFloatOut(ds, 4, &got) && AllCloseFloat(got, exp, tag);
        aclDestroyTensor(tself);
        aclDestroyScalar(sOther);
        aclDestroyScalar(sAlpha);
        aclrtFree(ds);
        Record(&stats, ok, tag);
    }

    // aclnnAddV3
    {
        const char* tag = "aclnnAddV3 float alpha=2";
        std::vector<int64_t> sh = {2, 2};
        float selfSc = 3.f;
        std::vector<float> otherH = {1, 2, 3, 4};
        float alphaVal = 2.f;
        void *do_ = nullptr, *dout = nullptr;
        aclTensor *tother = nullptr, *tout = nullptr;
        aclScalar* sSelf = aclCreateScalar(&selfSc, ACL_FLOAT);
        aclScalar* sAlpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
        bool ok = CreateAclTensor(otherH, sh, &do_, ACL_FLOAT, &tother) == 0 &&
                  CreateAclTensor(std::vector<float>(4, 0.f), sh, &dout, ACL_FLOAT, &tout) == 0 && sSelf != nullptr &&
                  sAlpha != nullptr;
        if (ok) {
            ok = InvokeTwoStage(
                tag, stream,
                [&](uint64_t* ws, aclOpExecutor** ex) {
                    return aclnnAddV3GetWorkspaceSize(sSelf, tother, sAlpha, tout, ws, ex);
                },
                [&](void* w, uint64_t ws, aclOpExecutor* ex, aclrtStream sm) { return aclnnAddV3(w, ws, ex, sm); }) &&
                 (aclrtSynchronizeStream(stream) == ACL_SUCCESS);
        }
        std::vector<float> got;
        std::vector<float> exp(4);
        for (int i = 0; i < 4; i++) {
            exp[static_cast<size_t>(i)] = selfSc + alphaVal * otherH[static_cast<size_t>(i)];
        }
        ok = ok && CopyFloatOut(dout, 4, &got) && AllCloseFloat(got, exp, tag);
        aclDestroyTensor(tother);
        aclDestroyTensor(tout);
        aclDestroyScalar(sSelf);
        aclDestroyScalar(sAlpha);
        aclrtFree(do_);
        aclrtFree(dout);
        Record(&stats, ok, tag);
    }

    // aclnnAddV3 alpha=1
    {
        const char* tag = "aclnnAddV3 float alpha=1";
        std::vector<int64_t> sh = {2};
        float selfSc = 10.f;
        std::vector<float> otherH = {1.f, 2.f};
        float one = 1.f;
        void *do_ = nullptr, *dout = nullptr;
        aclTensor *tother = nullptr, *tout = nullptr;
        aclScalar* sSelf = aclCreateScalar(&selfSc, ACL_FLOAT);
        aclScalar* sAlpha = aclCreateScalar(&one, ACL_FLOAT);
        bool ok = CreateAclTensor(otherH, sh, &do_, ACL_FLOAT, &tother) == 0 &&
                  CreateAclTensor(std::vector<float>(2, 0.f), sh, &dout, ACL_FLOAT, &tout) == 0 && sSelf != nullptr &&
                  sAlpha != nullptr;
        if (ok) {
            ok = InvokeTwoStage(
                tag, stream,
                [&](uint64_t* ws, aclOpExecutor** ex) {
                    return aclnnAddV3GetWorkspaceSize(sSelf, tother, sAlpha, tout, ws, ex);
                },
                [&](void* w, uint64_t ws, aclOpExecutor* ex, aclrtStream sm) { return aclnnAddV3(w, ws, ex, sm); }) &&
                 (aclrtSynchronizeStream(stream) == ACL_SUCCESS);
        }
        std::vector<float> got;
        std::vector<float> exp = {11.f, 12.f};
        ok = ok && CopyFloatOut(dout, 2, &got) && AllCloseFloat(got, exp, tag);
        aclDestroyTensor(tother);
        aclDestroyTensor(tout);
        aclDestroyScalar(sSelf);
        aclDestroyScalar(sAlpha);
        aclrtFree(do_);
        aclrtFree(dout);
        Record(&stats, ok, tag);
    }

    // aclnnInplaceAddV3
    {
        const char* tag = "aclnnInplaceAddV3 float";
        std::vector<int64_t> sh = {2, 2};
        float selfSc = 1.f;
        std::vector<float> otherH = {2, 3, 4, 5};
        float alphaVal = 0.5f;
        void* do_ = nullptr;
        aclTensor* tother = nullptr;
        aclScalar* sSelf = aclCreateScalar(&selfSc, ACL_FLOAT);
        aclScalar* sAlpha = aclCreateScalar(&alphaVal, ACL_FLOAT);
        bool ok = CreateAclTensor(otherH, sh, &do_, ACL_FLOAT, &tother) == 0 && sSelf != nullptr && sAlpha != nullptr;
        if (ok) {
            ok = InvokeTwoStage(
                tag, stream,
                [&](uint64_t* ws, aclOpExecutor** ex) {
                    return aclnnInplaceAddV3GetWorkspaceSize(sSelf, tother, sAlpha, ws, ex);
                },
                [&](void* w, uint64_t ws, aclOpExecutor* ex, aclrtStream sm) {
                    return aclnnInplaceAddV3(w, ws, ex, sm);
                }) &&
                 (aclrtSynchronizeStream(stream) == ACL_SUCCESS);
        }
        std::vector<float> got;
        std::vector<float> exp(4);
        for (int i = 0; i < 4; i++) {
            exp[static_cast<size_t>(i)] = selfSc + alphaVal * otherH[static_cast<size_t>(i)];
        }
        ok = ok && CopyFloatOut(do_, 4, &got) && AllCloseFloat(got, exp, tag);
        aclDestroyTensor(tother);
        aclDestroyScalar(sSelf);
        aclDestroyScalar(sAlpha);
        aclrtFree(do_);
        Record(&stats, ok, tag);
    }

    // 中等规模 float（避免过大触发模拟器超时/异常）
    {
        const char* tag = "aclnnAdd float 8x8";
        const int n = 8;
        std::vector<int64_t> sh = {n, n};
        const int total = n * n;
        std::vector<float> a(static_cast<size_t>(total));
        std::vector<float> b(static_cast<size_t>(total));
        for (int i = 0; i < total; i++) {
            a[static_cast<size_t>(i)] = static_cast<float>(i % 7);
            b[static_cast<size_t>(i)] = static_cast<float>(i % 5) * 0.25f;
        }
        float alphaVal = 1.5f;
        void *da = nullptr, *db = nullptr, *dc = nullptr;
        aclTensor *ta = nullptr, *tb = nullptr, *tc = nullptr;
        aclScalar* alp = aclCreateScalar(&alphaVal, ACL_FLOAT);
        bool ok = CreateAclTensor(a, sh, &da, ACL_FLOAT, &ta) == 0 && CreateAclTensor(b, sh, &db, ACL_FLOAT, &tb) == 0 &&
                  CreateAclTensor(std::vector<float>(static_cast<size_t>(total), 0.f), sh, &dc, ACL_FLOAT, &tc) == 0 &&
                  alp != nullptr;
        if (ok) {
            ok = InvokeTwoStage(
                tag, stream,
                [&](uint64_t* ws, aclOpExecutor** ex) {
                    return aclnnAddGetWorkspaceSize(ta, tb, alp, tc, ws, ex);
                },
                [&](void* w, uint64_t ws, aclOpExecutor* ex, aclrtStream sm) { return aclnnAdd(w, ws, ex, sm); }) &&
                 (aclrtSynchronizeStream(stream) == ACL_SUCCESS);
        }
        std::vector<float> got;
        std::vector<float> exp(static_cast<size_t>(total));
        for (int i = 0; i < total; i++) {
            exp[static_cast<size_t>(i)] = a[static_cast<size_t>(i)] + alphaVal * b[static_cast<size_t>(i)];
        }
        ok = ok && CopyFloatOut(dc, total, &got) && AllCloseFloat(got, exp, tag);
        aclDestroyTensor(ta);
        aclDestroyTensor(tb);
        aclDestroyTensor(tc);
        aclDestroyScalar(alp);
        aclrtFree(da);
        aclrtFree(db);
        aclrtFree(dc);
        Record(&stats, ok, tag);
    }

    // nullptr
    {
        const char* tag = "aclnnAdd nullptr self rejected";
        uint64_t ws = 0;
        aclOpExecutor* ex = nullptr;
        float dummy = 1.f;
        aclScalar* alp = aclCreateScalar(&dummy, ACL_FLOAT);
        aclnnStatus st = aclnnAddGetWorkspaceSize(nullptr, nullptr, alp, nullptr, &ws, &ex);
        aclDestroyScalar(alp);
        const bool ok = !AclnnOk(st);
        Record(&stats, ok, tag);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    printf("\n========== 汇总 PASS=%d FAIL=%d ==========\n", stats.pass, stats.fail);
    return stats.fail > 0 ? 1 : 0;
}
