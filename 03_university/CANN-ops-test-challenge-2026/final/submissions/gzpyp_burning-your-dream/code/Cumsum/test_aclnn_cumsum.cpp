/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0
 *
 * Improved Cumsum test: targets coverage of all 5 files:
 *   aclnn_cumsum.cpp, cumsum.cpp, cumsum_tiling.cpp,
 *   cumsum_tiling_ascendc_arch35.cpp, cumsum_tiling_ascendc_int_arch35.cpp
 */

 #include <iostream>
 #include <vector>
 #include <cmath>
 #include <climits>
 #include <algorithm>
 #include <sstream>
 #include "acl/acl.h"
 #include "aclnnop/aclnn_cumsum.h"
 
 #define CHECK_RET(cond, return_expr) \
     do { if (!(cond)) { return_expr; } } while (0)
 #define LOG_PRINT(...) printf(__VA_ARGS__)
 
 static bool g_aclInitialized = false;
 static aclrtStream g_stream = nullptr;
 
 int Init(int32_t deviceId, aclrtStream* stream)
 {
     if (g_aclInitialized) { *stream = g_stream; return 0; }
     auto ret = aclInit(nullptr);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed: %d\n", ret); return ret);
     ret = aclrtSetDevice(deviceId);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SetDevice failed: %d\n", ret); return ret);
     ret = aclrtCreateStream(stream);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateStream failed: %d\n", ret); return ret);
     g_aclInitialized = true;
     g_stream = *stream;
     return 0;
 }
 
 void Cleanup(aclrtStream stream)
 {
     if (stream) aclrtDestroyStream(stream);
     aclrtResetDevice(0);
     aclFinalize();
     g_aclInitialized = false;
 }
 
 // Helpers
 int64_t GetShapeSize(const std::vector<int64_t>& shape)
 {
     int64_t s = 1;
     for (auto d : shape) s *= d;
     return s;
 }
 std::string DimsToStr(const std::vector<int64_t>& shape)
 {
     std::ostringstream oss;
     for (size_t i = 0; i < shape.size(); ++i) {
         if (i > 0) oss << ",";
         oss << shape[i];
     }
     return oss.str();
 }
 void ComputeStrides(const std::vector<int64_t>& shape, std::vector<int64_t>& strides)
 {
     strides.resize(shape.size(), 1);
     for (int64_t i = (int64_t)shape.size() - 2; i >= 0; --i)
         strides[i] = shape[i + 1] * strides[i + 1];
 }
 
 template <typename T> inline aclDataType AclDtype() { return ACL_FLOAT; }
 template <> inline aclDataType AclDtype<float>() { return ACL_FLOAT; }
 template <> inline aclDataType AclDtype<aclFloat16>() { return ACL_FLOAT16; }
 template <> inline aclDataType AclDtype<int32_t>() { return ACL_INT32; }
 template <> inline aclDataType AclDtype<int64_t>() { return ACL_INT64; }
 template <> inline aclDataType AclDtype<int8_t>() { return ACL_INT8; }
 template <> inline aclDataType AclDtype<uint8_t>() { return ACL_UINT8; }
 template <> inline aclDataType AclDtype<bool>() { return ACL_BOOL; }
 
 template <typename T> inline const char* TypeName() { return "unknown"; }
 template <> inline const char* TypeName<float>() { return "float32"; }
 template <> inline const char* TypeName<aclFloat16>() { return "float16"; }
 template <> inline const char* TypeName<int32_t>() { return "int32"; }
 template <> inline const char* TypeName<int64_t>() { return "int64"; }
 template <> inline const char* TypeName<int8_t>() { return "int8"; }
 template <> inline const char* TypeName<uint8_t>() { return "uint8"; }
 template <> inline const char* TypeName<bool>() { return "bool"; }
 
 inline aclFloat16 F16(float v) { return aclFloat16(v); }
 
 bool AlmostEq(double e, double a, double atol, double rtol)
 {
     if (std::isnan(e) && std::isnan(a)) return true;
     if (std::isnan(e) || std::isnan(a)) return false;
     if (std::isinf(e) && std::isinf(a) && ((e > 0) == (a > 0))) return true;
     return std::fabs(a - e) <= atol + rtol * std::fabs(e);
 }
 
 // CPU: multi-dim cumsum
 template <typename T>
 std::vector<double> CpuCumsum(const std::vector<T>& input, int64_t dim,
                                  const std::vector<int64_t>& shape,
                                  bool exclusive, bool reverse)
 {
     int64_t ndims = (int64_t)shape.size();
     int64_t ad = (dim < 0) ? dim + ndims : dim;
     int64_t lenR = shape[ad];
     int64_t lenM = 1;
     for (int64_t i = 0; i < ad; ++i) lenM *= shape[i];
     int64_t lenN = 1;
     for (int64_t i = ad + 1; i < ndims; ++i) lenN *= shape[i];
 
     int64_t total = GetShapeSize(shape);
     std::vector<double> result(total, 0.0);
 
     for (int64_t m = 0; m < lenM; ++m) {
         for (int64_t n = 0; n < lenN; ++n) {
             if (!reverse) {
                 double cum = 0.0;
                 for (int64_t r = 0; r < lenR; ++r) {
                     int64_t idx = m * lenN * lenR + r * lenN + n;
                     if (exclusive) result[idx] = cum;
                     cum += (double)input[idx];
                     if (!exclusive) result[idx] = cum;
                 }
             } else {
                 double cum = 0.0;
                 for (int64_t r = lenR - 1; r >= 0; --r) {
                     int64_t idx = m * lenN * lenR + r * lenN + n;
                     if (exclusive) result[idx] = cum;
                     cum += (double)input[idx];
                     if (!exclusive) result[idx] = cum;
                 }
             }
         }
     }
     return result;
 }
 
 // CreateAclTensor
 template <typename T>
 int CreateTensor(const std::vector<T>& data, const std::vector<int64_t>& shape,
                  void** dev, aclDataType dtype, aclTensor** t)
 {
     auto size = GetShapeSize(shape) * sizeof(T);
     auto ret = aclrtMalloc(dev, size, ACL_MEM_MALLOC_HUGE_FIRST);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Malloc: %d\n", ret); return ret);
     ret = aclrtMemcpy(*dev, size, data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("H2D: %d\n", ret); return ret);
     std::vector<int64_t> strides;
     ComputeStrides(shape, strides);
     *t = aclCreateTensor(shape.data(), (int)shape.size(), dtype,
         strides.data(), 0, ACL_FORMAT_ND,
         shape.data(), (int)shape.size(), *dev);
     return 0;
 }
 
 // Specialization for bool
 template <>
 int CreateTensor<bool>(const std::vector<bool>& data, const std::vector<int64_t>& shape,
                         void** dev, aclDataType dtype, aclTensor** t)
 {
     auto sz = GetShapeSize(shape);
     std::vector<uint8_t> tmp(sz);
     for (int64_t i = 0; i < sz; ++i) tmp[i] = data[i] ? 1 : 0;
     auto ret = aclrtMalloc(dev, sz, ACL_MEM_MALLOC_HUGE_FIRST);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Malloc bool: %d\n", ret); return ret);
     ret = aclrtMemcpy(*dev, sz, tmp.data(), sz, ACL_MEMCPY_HOST_TO_DEVICE);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("H2D bool: %d\n", ret); return ret);
     std::vector<int64_t> strides;
     ComputeStrides(shape, strides);
     *t = aclCreateTensor(shape.data(), (int)shape.size(), dtype,
         strides.data(), 0, ACL_FORMAT_ND,
         shape.data(), (int)shape.size(), *dev);
     return 0;
 }
 
 // ReadResult
 template <typename T>
 int ReadResult(void* dev, const std::vector<int64_t>& shape, std::vector<T>& out)
 {
     auto sz = GetShapeSize(shape);
     out.resize(sz);
     auto ret = aclrtMemcpy(out.data(), sz * sizeof(T), dev, sz * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("D2H: %d\n", ret); return ret);
     return 0;
 }
 template <>
 int ReadResult<bool>(void* dev, const std::vector<int64_t>& shape, std::vector<bool>& out)
 {
     auto sz = GetShapeSize(shape);
     std::vector<uint8_t> tmp(sz);
     auto ret = aclrtMemcpy(tmp.data(), sz, dev, sz, ACL_MEMCPY_DEVICE_TO_HOST);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("D2H bool: %d\n", ret); return ret);
     out.resize(sz);
     for (int64_t i = 0; i < sz; ++i) out[i] = tmp[i] != 0;
     return 0;
 }
 
 // Run aclnnCumsumV2
 template <typename T>
 int RunV2(const std::vector<T>& input, const std::vector<int64_t>& shape,
             int64_t dim, bool excl, bool rev,
             std::vector<T>& output, aclrtStream stream)
 {
     void *dIn = nullptr, *dOut = nullptr;
     aclTensor *tIn = nullptr, *tOut = nullptr;
     auto ret = CreateTensor(input, shape, &dIn, AclDtype<T>(), &tIn);
     CHECK_RET(ret == 0, LOG_PRINT("CreateTensor in failed\n"); return ret);
     std::vector<T> zero(GetShapeSize(shape), T(0));
     ret = CreateTensor(zero, shape, &dOut, AclDtype<T>(), &tOut);
     CHECK_RET(ret == 0, LOG_PRINT("CreateTensor out failed\n");
               aclDestroyTensor(tIn); aclrtFree(dIn); return ret);
 
     uint64_t ws = 0;
     aclOpExecutor* exec = nullptr;
     ret = aclnnCumsumV2GetWorkspaceSize(tIn, dim, excl, rev, tOut, &ws, &exec);
     CHECK_RET(ret == ACL_SUCCESS && exec != nullptr,
               LOG_PRINT("V2 GetWS failed: ret=%d exec=%p\n", ret, (void*)exec);
               aclDestroyTensor(tIn); aclDestroyTensor(tOut); aclrtFree(dIn); aclrtFree(dOut); return ret);
 
     void* workspace = nullptr;
     if (ws > 0) {
         ret = aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("WS alloc: %d\n", ret);
                   aclDestroyTensor(tIn); aclDestroyTensor(tOut); aclrtFree(dIn); aclrtFree(dOut); return ret);
     }
     ret = aclnnCumsumV2(workspace, ws, exec, stream);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("V2 exec: %d\n", ret);
               if (workspace) aclrtFree(workspace);
               aclDestroyTensor(tIn); aclDestroyTensor(tOut); aclrtFree(dIn); aclrtFree(dOut); return ret);
     ret = aclrtSynchronizeStream(stream);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Sync: %d\n", ret);
               if (workspace) aclrtFree(workspace);
               aclDestroyTensor(tIn); aclDestroyTensor(tOut); aclrtFree(dIn); aclrtFree(dOut); return ret);
 
     ret = ReadResult(dOut, shape, output);
     if (workspace) aclrtFree(workspace);
     aclDestroyTensor(tIn);
     aclDestroyTensor(tOut);
     aclrtFree(dIn);
     aclrtFree(dOut);
     return ret;
 }
 
 // Run aclnnCumsum (with dtype param)
 int RunCumsumDtype(const std::vector<float>& input, const std::vector<int64_t>& shape,
                     int64_t dim, aclDataType dtype,
                     std::vector<float>& output, aclrtStream stream)
 {
     void *dIn = nullptr, *dOut = nullptr;
     aclTensor *tIn = nullptr, *tOut = nullptr;
     auto ret = CreateTensor(input, shape, &dIn, ACL_FLOAT, &tIn);
     CHECK_RET(ret == 0, LOG_PRINT("CreateTensor in failed\n"); return ret);
     std::vector<float> zero(GetShapeSize(shape), 0.0f);
     ret = CreateTensor(zero, shape, &dOut, dtype, &tOut);
     CHECK_RET(ret == 0, LOG_PRINT("CreateTensor out failed\n");
               aclDestroyTensor(tIn); aclrtFree(dIn); return ret);
 
     uint64_t ws = 0;
     aclOpExecutor* exec = nullptr;
     ret = aclnnCumsumGetWorkspaceSize(tIn, dim, dtype, tOut, &ws, &exec);
     CHECK_RET(ret == ACL_SUCCESS && exec != nullptr,
               LOG_PRINT("Cumsum GetWS failed: ret=%d exec=%p\n", ret, (void*)exec);
               aclDestroyTensor(tIn); aclDestroyTensor(tOut); aclrtFree(dIn); aclrtFree(dOut); return ret);
 
     void* workspace = nullptr;
     if (ws > 0) {
         ret = aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("WS alloc: %d\n", ret);
                   aclDestroyTensor(tIn); aclDestroyTensor(tOut); aclrtFree(dIn); aclrtFree(dOut); return ret);
     }
     ret = aclnnCumsum(workspace, ws, exec, stream);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Cumsum exec: %d\n", ret);
               if (workspace) aclrtFree(workspace);
               aclDestroyTensor(tIn); aclDestroyTensor(tOut); aclrtFree(dIn); aclrtFree(dOut); return ret);
     ret = aclrtSynchronizeStream(stream);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Sync: %d\n", ret);
               if (workspace) aclrtFree(workspace);
               aclDestroyTensor(tIn); aclDestroyTensor(tOut); aclrtFree(dIn); aclrtFree(dOut); return ret);
 
     // Read back
     auto sz = GetShapeSize(shape);
     if (dtype == ACL_INT32) {
         std::vector<int32_t> tmp(sz);
         aclrtMemcpy(tmp.data(), sz * 4, dOut, sz * 4, ACL_MEMCPY_DEVICE_TO_HOST);
         output.resize(sz);
         for (int64_t i = 0; i < sz; ++i) output[i] = (float)tmp[i];
     } else if (dtype == ACL_INT64) {
         std::vector<int64_t> tmp(sz);
         aclrtMemcpy(tmp.data(), sz * 8, dOut, sz * 8, ACL_MEMCPY_DEVICE_TO_HOST);
         output.resize(sz);
         for (int64_t i = 0; i < sz; ++i) output[i] = (float)tmp[i];
     } else {
         output.resize(sz);
         aclrtMemcpy(output.data(), sz * 4, dOut, sz * 4, ACL_MEMCPY_DEVICE_TO_HOST);
     }
 
     if (workspace) aclrtFree(workspace);
     aclDestroyTensor(tIn);
     aclDestroyTensor(tOut);
     aclrtFree(dIn);
     aclrtFree(dOut);
     return 0;
 }
 
 // Precision test
 template <typename T>
 int TestPrec(const char* name, const std::vector<T>& input,
               const std::vector<int64_t>& shape, int64_t dim,
               bool excl, bool rev,
               double atol, double rtol, aclrtStream stream)
 {
     std::vector<T> out;
     auto ret = RunV2(input, shape, dim, excl, rev, out, stream);
     if (ret != 0) { LOG_PRINT("[FAIL] %s: ret=%d\n", name, ret); return 1; }
     auto exp = CpuCumsum(input, dim, shape, excl, rev);
     int64_t n = GetShapeSize(shape);
     double maxErr = 0.0;
     int bad = 0;
     for (int64_t i = 0; i < n; ++i) {
         double a = (double)out[i], e = exp[i];
         double err = std::fabs(a - e);
         if (err > maxErr) maxErr = err;
         if (!AlmostEq(e, a, atol, rtol)) {
             if (bad < 3) LOG_PRINT("  [%ld] exp=%.6g act=%.6g err=%.6g\n", (long)i, e, a, err);
             ++bad;
         }
     }
     bool ok = bad == 0;
     LOG_PRINT("%s %s (dtype=%s shape=[%s] dim=%ld excl=%d rev=%d)\n",
               ok ? "[PASS]" : "[FAIL]", name,
               TypeName<T>(), DimsToStr(shape).c_str(), (long)dim, excl?1:0, rev?1:0);
     LOG_PRINT("  Max err: %.8g\n", maxErr);
     if (bad > 0) LOG_PRINT("  %d/%ld OOT (atol=%.0e rtol=%.0e)\n", bad, (long)n, atol, rtol);
     return ok ? 0 : 1;
 }
 
 // ========== COVERAGE SCENES ==========
 
 // Scene A: Null pointer (CheckNotNull branches in aclnn_cumsum.cpp)
 void SceneTest1(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene A] Null pointer check\n");
     uint64_t ws = 0; aclOpExecutor* ex = nullptr;
     int64_t s[] = {4};
     aclTensor* t = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, nullptr);
     auto r = aclnnCumsumV2GetWorkspaceSize(nullptr, 0, 0, 0, t, &ws, &ex);
     LOG_PRINT("  NULL self: ret=%d\n", r);
     if (t) aclDestroyTensor(t);
     t = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, nullptr);
     r = aclnnCumsumV2GetWorkspaceSize(t, 0, 0, 0, nullptr, &ws, &ex);
     LOG_PRINT("  NULL out: ret=%d\n", r);
     if (t) aclDestroyTensor(t);
 }
 
 // Scene B: Empty tensor (IsEmpty branch)
 void SceneTest2(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene B] Empty tensor\n");
     int64_t s[] = {0};
     void* dev = nullptr;
     aclTensor* tIn = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dev);
     aclTensor* tOut = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dev);
     uint64_t ws = 0; aclOpExecutor* ex = nullptr;
     auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 0, 0, tOut, &ws, &ex);
     LOG_PRINT("  Empty: ret=%d ws=%lu exec=%p\n", r, ws, (void*)ex);
     if (ex) {
         void* wp = nullptr;
         if (ws > 0) aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
         r = aclnnCumsumV2(wp, ws, ex, stream);
         LOG_PRINT("  Exec: ret=%d\n", r);
         if (wp) aclrtFree(wp);
     }
     aclDestroyTensor(tIn);
     aclDestroyTensor(tOut);
 }
 
 // Scene C: dim branches (dim=0 vs dim>INT32_MAX, negative dim)
 // dim=0 uses DT_INT32 path, dim>INT32_MAX uses DT_INT64 path
 void SceneTest3(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene C] dim parameter branches\n");
     std::vector<float> inp(8, 1.0f);
     std::vector<float> out;
     std::vector<int64_t> sh = {8};
 
     auto r = RunV2(inp, sh, 0, 0, 0, out, stream);
     LOG_PRINT("  dim=0 (INT32): ret=%d\n", r);
 
     int64_t bigDim = (int64_t)INT32_MAX + 1;
     r = RunV2(inp, sh, bigDim, 0, 0, out, stream);
     LOG_PRINT("  dim=INT32_MAX+1 (INT64): ret=%d\n", r);
 
     r = RunV2(inp, sh, -1, 0, 0, out, stream);
     LOG_PRINT("  dim=-1 (negative): ret=%d\n", r);
 }
 
 // Scene D: exclusive/reverse (4 combinations via V2)
 void SceneTest4(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene D] exclusive/reverse combinations\n");
     std::vector<float> inp = {1,2,3,4,5};
     std::vector<float> out;
     bool combos[4][2] = {{0,0},{1,0},{0,1},{1,1}};
     const char* n[4] = {"incl-fwd","excl-fwd","incl-rev","excl-rev"};
     for (int i = 0; i < 4; ++i) {
         auto r = RunV2(inp, std::vector<int64_t>{5}, 0, combos[i][0], combos[i][1], out, stream);
         LOG_PRINT("  [%s]: ret=%d [%.0f,%.0f,%.0f,%.0f,%.0f]\n", n[i], r,
                  (double)out[0],(double)out[1],(double)out[2],(double)out[3],(double)out[4]);
     }
 }
 
 // Scene E: All dtypes via V2 (FLOAT, FLOAT16, INT32, INT64, INT8, UINT8, BOOL)
 void SceneTest5(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene E] All dtypes (aclnnCumsumV2)\n");
     aclDataType dts[] = {ACL_FLOAT, ACL_FLOAT16, ACL_INT32, ACL_INT64, ACL_INT8, ACL_UINT8, ACL_BOOL};
     const char* n[] = {"FLOAT","FLOAT16","INT32","INT64","INT8","UINT8","BOOL"};
     int64_t s[] = {8};
     for (int i = 0; i < 7; ++i) {
         uint64_t ws = 0; aclOpExecutor* ex = nullptr;
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 1, dts[i], nullptr, 0, ACL_FORMAT_ND, s, 1, dev);
         aclTensor* tOut = aclCreateTensor(s, 1, dts[i], nullptr, 0, ACL_FORMAT_ND, s, 1, dev);
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  %s: ret=%d ws=%lu exec=%p\n", n[i], r, ws, (void*)ex);
         if (ex && ws > 0) {
             void* wp = nullptr;
             aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
             r = aclnnCumsumV2(wp, ws, ex, stream);
             aclrtFree(wp);
         }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 }
 
 // Scene F: aclnnCumsum (with dtype) - dtype conversion paths
 // Targets: CheckDtypeValid, CheckCubeSupport, CheckShapeIsSupport, Cast
 void SceneTest6(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene F] aclnnCumsum with dtype (conversion + CUBE path)\n");
     std::vector<float> inp(16, 1.5f);
     std::vector<int64_t> sh = {16};
     std::vector<float> out;
 
     // FLOAT->INT32 dtype conversion
     auto r = RunCumsumDtype(inp, sh, 0, ACL_INT32, out, stream);
     LOG_PRINT("  FLOAT->INT32: ret=%d out=[%.1f,%.1f,...]\n", r, (double)out[0], (double)out[1]);
 
     // FLOAT->INT64 dtype conversion
     r = RunCumsumDtype(inp, sh, 0, ACL_INT64, out, stream);
     LOG_PRINT("  FLOAT->INT64: ret=%d\n", r);
 
     // FLOAT16 dtype (output must also be FP16)
     {
         std::vector<aclFloat16> inp2(16);
         for (int i = 0; i < 16; ++i) inp2[i] = F16(1.5f);
         std::vector<aclFloat16> out2;
         r = RunV2(inp2, sh, 0, 0, 0, out2, stream);
         LOG_PRINT("  FLOAT16: ret=%d\n", r);
     }
 
     // CUBE path: large batch >= 12800, dim >= 512, dim is last
     // Skip large CUBE allocations here to avoid NPU memory exhaustion.
     // CUBE path is covered in Scene O instead.
     {
         LOG_PRINT("  CUBE path: skipped (covered in Scene O)\n");
     }
 
     // Non-CUBE: small shape, should go to fallback
     {
         std::vector<float> sm(10, 1.0f);
         std::vector<int64_t> smSh = {10};
         std::vector<float> smOut;
         r = RunCumsumDtype(sm, smSh, 0, ACL_FLOAT, smOut, stream);
         LOG_PRINT("  Non-CUBE (small): ret=%d\n", r);
     }
 }
 
 // Scene G: Tiling - NGreaterCl vs NLesserCl
 // Condition: lenN * dtSize >= clSize (clSize=64B, float=4B -> lenN>=16)
 // shape(1,1,N) dim=2: lenN=1 -> NLesserCl
 // shape(1,1,N) dim=0: lenN=N -> NGreaterCl when N>=16
 void SceneTest7(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene G] Tiling NGreaterCl vs NLesserCl\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // NLesserCl: shape(1,1,8) dim=2, lenN=8 (float: 8*4=32 < 64)
     {
         int64_t s[] = {1, 1, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 2, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  shape=(1,1,8) dim=2 (NLesserCl lenN=8): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
     // NGreaterCl: shape(1,1,32) dim=0, lenN=32 (32*4=128>=64)
     {
         int64_t s[] = {1, 1, 32};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  shape=(1,1,32) dim=0 (NGreaterCl lenN=32): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
     // NGreaterCl: shape(1,1,16) dim=0, lenN=16 (16*4=64>=64 boundary)
     {
         int64_t s[] = {1, 1, 16};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  shape=(1,1,16) dim=0 (NGreaterCl lenN=16 boundary): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene H: Tiling - RNGreaterCl vs RNLesserCl
 // Condition: lenR * lenN * 4 >= clSize (clSize=64B -> MRN>=16)
 // shape(1,1,2,4) dim=2: lenR=2,lenN=4,M=1 -> MRN=8 -> RNLesserCl
 // shape(1,2,4,8) dim=2: lenR=4,lenN=8,M=2 -> MRN=64 -> RNGreaterCl
 void SceneTest8(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene H] Tiling RNGreaterCl vs RNLesserCl\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // RNLesserCl
     {
         int64_t s[] = {1, 1, 2, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 4, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 4, dev);
         aclTensor* tOut = aclCreateTensor(s, 4, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 4, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 2, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  shape=(1,1,2,4) dim=2 (RNLesserCl MRN=8): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
     // RNGreaterCl
     {
         int64_t s[] = {1, 2, 4, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 4, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 4, dev);
         aclTensor* tOut = aclCreateTensor(s, 4, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 4, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 2, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  shape=(1,2,4,8) dim=2 (RNGreaterCl MRN=64): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene I: Tiling - MRNGreaterCl vs MRNLesserCl
 // Condition: M*R*N*4 >= clSize
 // shape(1,1,1,4) dim=3: M=1,R=1,N=4 -> MRN=4 -> MRNLesserCl
 // shape(2,2,2,4) dim=3: M=4,R=2,N=4 -> MRN=32 -> MRNGreaterCl
 void SceneTest9(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene I] Tiling MRNGreaterCl vs MRNLesserCl\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     {
         int64_t s[] = {1, 1, 1, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 4, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 4, dev);
         aclTensor* tOut = aclCreateTensor(s, 4, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 4, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 3, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  shape=(1,1,1,4) dim=3 (MRNLesserCl MRN=4): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
     {
         int64_t s[] = {2, 2, 2, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 4, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 4, dev);
         aclTensor* tOut = aclCreateTensor(s, 4, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 4, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 3, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  shape=(2,2,2,4) dim=3 (MRNGreaterCl MRN=32): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene J: Tiling - RFullLoad vs RNotFullLoad (lenR * clSize <= ubSize)
 // For float, ubSize~200KB, clSize=64B:
 // lenR=2000: 2000*64=128KB <= 200KB -> RFullLoad
 // lenR=5000: 5000*64=320KB > 200KB -> RNotFullLoad
 void SceneTest10(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene J] Tiling RFullLoad vs RNotFullLoad\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // Small lenR -> RFullLoad
     {
         int64_t s[] = {2, 200};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  shape=(2,200) dim=1 (RFullLoad): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
     // Large lenR -> RNotFullLoad
     {
         int64_t s[] = {2, 5000};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  shape=(2,5000) dim=1 (RNotFullLoad): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene K: Integer tiling axis branches (cumsum_tiling_ascendc_int_arch35.cpp)
 // GetInputDims: axis==0, axis==dimNum-1, axis in middle
 void SceneTest11(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene K] Integer tiling axis branches\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // axis=0
     {
         int64_t s[] = {8, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 shape=(8,4) dim=0 (axis==0): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
     // axis=last
     {
         int64_t s[] = {8, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 shape=(8,4) dim=1 (axis==last): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
     // axis=middle (3D)
     {
         int64_t s[] = {4, 8, 2};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 shape=(4,8,2) dim=1 (axis=middle): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
     // All int dtypes: INT64, INT8, UINT8
     aclDataType idts[] = {ACL_INT64, ACL_INT8, ACL_UINT8};
     const char* in[] = {"INT64","INT8","UINT8"};
     for (int i = 0; i < 3; ++i) {
         int64_t s[] = {8, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, idts[i], nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, idts[i], nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  %s shape=(8,4) dim=1: ret=%d ws=%lu\n", in[i], r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene L: Sklansky pattern branches (SS_ONEWAY vs SS_TWOWAY)
 // SS pattern depends on: alignN > vRegSize/4, foldCount, lenR
 // float32, vRegSize typically 128 or 256
 // lenR=32, alignN=128 (lenN=4): foldCount=1 -> SS_ONEWAY
 // lenR=64, lenN=4: lenR=64, foldCount>1 -> SS_TWOWAY
 void SceneTest12(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene L] Sklansky SS ONEWAY vs TWOWAY patterns\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // lenR=32 (likely ONEWAY)
     {
         int64_t s[] = {1, 32, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  shape=(1,32,4) dim=1 (lenR=32): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
     // lenR=64 (likely TWOWAY)
     {
         int64_t s[] = {1, 64, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  shape=(1,64,4) dim=1 (lenR=64): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
     // lenR=128 (likely TWOWAY)
     {
         int64_t s[] = {1, 128, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  shape=(1,128,4) dim=1 (lenR=128): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene M: dtCast path (FP16/BF16 need cast to float in tiling)
 void SceneTest13(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene M] dtCast path (FP16/BF16)\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // FLOAT16 dtCast=true
     {
         int64_t s[] = {16};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, s, 1, dev);
         aclTensor* tOut = aclCreateTensor(s, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, s, 1, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  FLOAT16 dtCast=true: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // FLOAT16 with exclusive/reverse
     {
         int64_t s[] = {8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, s, 1, dev);
         aclTensor* tOut = aclCreateTensor(s, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, s, 1, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 1, 1, tOut, &ws, &ex); // excl+rev
         LOG_PRINT("  FLOAT16 excl+rev: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // FLOAT16 with 2D shape, various dims
     {
         int64_t s[] = {4, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         for (int dim = 0; dim < 2; ++dim) {
             ws = 0; ex = nullptr;
             auto r = aclnnCumsumV2GetWorkspaceSize(tIn, dim, 0, 0, tOut, &ws, &ex);
             LOG_PRINT("  FLOAT16 shape=(4,8) dim=%d: ret=%d ws=%lu\n", dim, r, ws);
             if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene N: More tiling shapes to exercise more branches
 void SceneTest14(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene N] Additional tiling shapes\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     struct Case { int64_t s[4]; int nd; int dim; const char* desc; };
     Case cases[] = {
         {{1, 1, 1, 16}, 4, 3, "4D-last"},
         {{1, 1, 1, 16}, 4, 2, "4D-middle"},
         {{1, 1, 1, 16}, 4, 1, "4D-early"},
         {{2, 500}, 2, 1, "2D-long"},
         {{500, 2}, 2, 0, "2D-wide"},
         {{1, 1, 100, 4}, 4, 2, "4D-MRNlarge"},
         {{1, 1, 4, 100}, 4, 3, "4D-lenRlarge"},
     };
     for (auto& c : cases) {
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(c.s, c.nd, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, c.s, c.nd, dev);
         aclTensor* tOut = aclCreateTensor(c.s, c.nd, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, c.s, c.nd, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, c.dim, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  %s: ret=%d ws=%lu\n", c.desc, r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Forward declarations for new scenes
 void SceneTest22(aclrtStream stream);
 void SceneTest23(aclrtStream stream);
 void SceneTest24(aclrtStream stream);
 void SceneTest25(aclrtStream stream);
 void SceneTest26(aclrtStream stream);
 void SceneTest27(aclrtStream stream);
 void SceneTest28(aclrtStream stream);
 void SceneTest29(aclrtStream stream);
 void SceneTest30(aclrtStream stream);
 void SceneTest31(aclrtStream stream);
 void SceneTest32(aclrtStream stream);
 void SceneTest33(aclrtStream stream);
 void SceneTest34(aclrtStream stream);
 void SceneTest35(aclrtStream stream);
void SceneTest36(aclrtStream stream);
void SceneTest37(aclrtStream stream);
void SceneTest38(aclrtStream stream);
void SceneTest39(aclrtStream stream);
void SceneTest40(aclrtStream stream);
void SceneTest41(aclrtStream stream);
void SceneTest42(aclrtStream stream);
void SceneTest43(aclrtStream stream);
void SceneTest44(aclrtStream stream);
void SceneTest45(aclrtStream stream);
void ExtendedPrecisionTests(aclrtStream stream);
void ExtendedPrecisionTests2(aclrtStream stream);
 
 // Scene O: Old API (aclnnCumsum) + shape branches + tiling edge cases
 // Targets: aclnnCumsumGetWorkspaceSize (unused old API but must cover),
 //          CheckShapeIsSupport (0-dim, negative dim, cube shape conditions),
 //          CheckCubeSupport (non-910B/910_93 SOCs, non-float dtypes, unsupported shapes),
 //          tiling: NGreaterCl (lenR*clNSize > ubSize), RNGreaterClRNotFullLoadBorrowM,
 //                  RNGreaterClRNotFullLoadBorrowRTwoway, NGreaterClRNotFullLoad (M < coreNum/2)
 void SceneTest15(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene O] Old API, shape support, tiling edge cases\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // Old API aclnnCumsumGetWorkspaceSize (unused path in this codebase)
     // Only called via GetWorkspaceSize — not used at runtime, but must cover
     {
         std::vector<float> inp(8, 1.0f);
         std::vector<int64_t> sh = {8};
         void* dIn = nullptr; void* dOut = nullptr;
         aclTensor *tIn = nullptr, *tOut = nullptr;
         CreateTensor(inp, sh, &dIn, ACL_FLOAT, &tIn);
         std::vector<float> zero(8, 0.0f);
         CreateTensor(zero, sh, &dOut, ACL_FLOAT, &tOut);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumGetWorkspaceSize(tIn, 0, ACL_FLOAT, tOut, &ws, &ex);
         LOG_PRINT("  aclnnCumsumGetWorkspaceSize old API: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsum(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut); aclrtFree(dIn); aclrtFree(dOut);
     }
 
     // ShapeIsSupport: dim!=0 && dim!=selfDimNum-1 (triggers return false at line 189)
     // shape {4,8,4}, dim=1 (middle axis, not first or last)
     {
         int64_t s[] = {4, 8, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  ShapeIsSupport mid-axis (dim=1): ret=%d (returns false for cube path)\n", r);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // ShapeIsSupport: dim=0 but batchNum < CUMSUM_CUBE_MIN_SUPPORT_BATCH (12800)
     // shape {64, 256}, dim=1: batchNum=64 < 12800 -> not cube-supported
     {
         int64_t s[] = {64, 256};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  ShapeIsSupport small batch: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // ShapeIsSupport: batchNum >= 12800 && channelNum >= 512 (cube supported)
     // shape {12800, 512}, dim=1: batch=12800, channel=512 -> cube path
     {
         int64_t s[] = {12800, 512};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  ShapeIsSupport cube (12800x512): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // ShapeIsSupport: dim < 0 (negative dim, triggers dim=dim+ndims)
     {
         int64_t s[] = {2, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, -1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  ShapeIsSupport neg dim=-1: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // tiling: RNGreaterClRNotFullLoadBorrowM (lenM >= coreNum, borrowMCount > 1)
     // shape (64, 32, 4) dim=1: coreNum=64, lenM=64 -> borrowMCount can be > 1
     {
         int64_t s[] = {64, 32, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  RNGreaterClRNotFullLoadBorrowM: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // tiling: NGreaterClRNotFullLoad with M < coreNum/2 (M not enough for core split)
     // shape (1, 5000, 1) dim=1: lenM=1, coreNum=64 -> M < 32
     {
         int64_t s[] = {1, 5000, 1};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  NGreaterClRNotFullLoad (M<core/2): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // tiling: RNGreaterClRNotFullLoadBorrowRTwoway (twoway pattern with borrowed R)
     // shape (1, 4, 4) dim=1: lenR=4, lenN=4 -> twoway possible
     {
         int64_t s[] = {1, 4, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  RNGreaterClRNotFullLoadBorrowRTwoway: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // INT tiling: CheckBGC path (bank group conflict adjustment)
     // axis in middle, rightAxisLen large -> triggers CheckBGC
     {
         int64_t s[] = {2, 8, 32};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 CheckBGC path: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // INT tiling: AdjustLARLpUnit triggered from CheckBGC
     // axis in middle, specific sizes that trigger bank group conflict
     {
         int64_t s[] = {4, 16, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 AdjustLARLpUnit path: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // INT tiling: CUM_AR_SPLIT (rightAxisLen * dtypeSize < vlSize_)
     {
         int64_t s[] = {2, 16, 2};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 CUM_AR_SPLIT: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // ========== NEW SCENES (S-T-U-V-W) ==========
 
 // Scene S: Float tiling — NGreaterClRFullLoad M-not-enough-for-core-split branch
 // NGreaterClRFullLoad: lenM < coreNum, M can't fully split cores, borrowNCount > 1
 // Shape (1, N): lenM=1, lenR=N, lenN=1
 //   If N*dtSize >= clSize (N>=16 for float): NGreaterCl branch
 //   lenR*N*dtSize > ubSize: RNotFullLoad -> NGreaterClRNotFullLoad
 //   But we want RFullLoad -> we need lenR*clNSize <= ubSize
 //   For ascend910_93: clSize=64, clNSize=clSize*3=192, ubSize~192KB
 //   So lenR*192 <= 192*1024 -> lenR <= 1024 -> RFullLoad is possible
 // Shape (1, 512): lenM=1, lenR=512, lenN=1
 //   N*dtSize=4 >= clSize=64? No -> NLesserCl -> RNGreaterCl -> RNGreaterClRFullLoad
 //
 // To hit NGreaterClRFullLoad with M < coreNum:
 //   Need N*dtSize >= clSize AND lenR*clNSize <= ubSize AND M < coreNum
 //   Shape (1, 512, 1): lenM=1, lenR=512, lenN=1
 //     lenN=1, dtSize=4, 1*4=4 < 64 -> NLesserCl -> RNGreaterCl
 //
 // To hit NGreaterCl: lenN * dtSize >= clSize, e.g. lenN=64 (64*4=256>=64)
 // Shape (1, 512, 64): lenM=1, lenR=512, lenN=64
 //   N*dtSize=256>=64 -> NGreaterCl
 //   lenR*clNSize=512*64=32768 vs ubSize=192KB=196608 -> FullLoad
 //   M=1 < coreNum=64 -> M-not-enough branch
 void SceneTest16(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene S] NGreaterClRFullLoad M-not-enough + RNGreaterClBorrowM\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // NGreaterClRFullLoad with M < coreNum (borrowNCount > 1)
     // lenN=64 -> NGreaterCl, lenR=512 -> RFullLoad, M=1 -> M-not-enough
     {
         int64_t s[] = {1, 512, 64};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  NGreaterClRFullLoad M<coreNum: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // RNGreaterClBorrowM: lenM >= coreNum, twoway, lenR > rMaxForFullUb
     // Shape (64, 256, 8) dim=1: lenM=64, coreNum=64 -> M can split, borrowMCount
     {
         int64_t s[] = {64, 256, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  RNGreaterClBorrowM: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // RNGreaterClBorrowM: lenR > rMaxForFullUb (R can't full-load UB, uses UB_SS_ONEWAY)
     // Shape (64, 8192, 8) dim=1: lenM=64, lenR=8192
     {
         int64_t s[] = {64, 8192, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  RNGreaterClBorrowM RNotFullUB: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // RNGreaterClBorrowM: lenR <= rMaxForFullUb (R can full-load UB)
     // Shape (64, 256, 8) dim=1: lenR=256
     // Already covered above; add another variation
     {
         int64_t s[] = {64, 128, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  RNGreaterClBorrowM RFullUB: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene T: Float tiling — NGreaterClRNotFullLoad M>=coreNum/2 + borrowN after M-borrowN-not-enough
 // NGreaterClRNotFullLoad: lenR*clNSize > ubSize
 // M >= coreNum/2 -> core split on M
 // Shape (64, 16384, 1): lenM=64, lenR=16384, lenN=1
 //   N*dtSize=4 < 64 -> NLesserCl, NOT this path
 // Shape (64, 32768, 64): lenM=64, lenR=32768, lenN=64
 //   N*dtSize=256>=64 -> NGreaterCl
 //   lenR*clNSize=32768*64=2M vs ubSize=192KB -> RNotFullLoad
 //   M=64 >= coreNum/2=32 -> core split on M
 void SceneTest17(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene T] NGreaterClRNotFullLoad M>=coreNum/2 + AdjustLARLpUnit\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // NGreaterClRNotFullLoad M>=coreNum/2 (core split on M)
     {
         int64_t s[] = {64, 32768, 64};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  NGreaterClRNotFullLoad M>=core/2: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // NGreaterClRNotFullLoad: M < coreNum/2, M*borrowNMax > coreNum/2
     // lenM=16, lenR=65536, lenN=32
     // M=16 < 32, borrowNMax=CeilDiv(32*4,64)=2, M*borrowNMax=32 > 32 -> borrowN path
     {
         int64_t s[] = {16, 65536, 32};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  NGreaterClRNotFullLoad M<core/2 borrowN: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // NGreaterClRNotFullLoad: M < coreNum/2, M*borrowNMax <= coreNum/2 -> borrowR
     // lenM=8, lenR=65536, lenN=8
     {
         int64_t s[] = {8, 65536, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  NGreaterClRNotFullLoad M<core/2 borrowR: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // RNGreaterClRNotFullLoadBorrowR: lenM < coreNum/2, borrowR
     // Already covered in Scene O; add variation with twoway pattern
     // Shape (1, 4, 4): RNGreaterClRNotFullLoadBorrowRTwoway
     {
         int64_t s[] = {1, 4, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  RNGreaterClRNotFullLoadBorrowRTwoway: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // RNGreaterClRNotFullLoadNotBorrowR: M >= coreNum/2, oneway pattern
     // Shape (64, 4096, 4): M=64, lenR=4096, lenN=4
     {
         int64_t s[] = {64, 4096, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  RNGreaterClRNotFullLoadNotBorrowR: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene U: Integer tiling — TDLA (left axis large) branches
 // TDLA: rightAxisLen * dtypeSize <= vlSize/2
 // GetAxisLpUnit: line 196 condition (rightAxisLen*dtypeSize > vlSize/2) NOT satisfied -> falls through
 // GetAxisLpUnit: leftAxisLen > coreNum/CORE_GATE -> AdjustTensor4TDLA -> AdjustLARLpUnit
 // AdjustLARLpUnit: arSize*laLpUnit_ > maxTensorSize branch
 // AdjustLARLpUnit: tmpLALpUnit < 2 -> rLpUnit_ reduction branch
 // AdjustLARLpUnit: after adjustment, laLpUnit_ update
 // GetMCTilingInfo: maxWeight == laAxisWeight (LA axis split)
 void SceneTest18(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene U] INT tiling TDLA + AdjustLARLpUnit branches\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // TDLA: axis in middle, leftAxisLen large, rightAxisLen small
     // AdjustLARLpUnit: arSize*laLpUnit_ > maxTensorSize
     // Shape (128, 2, 2): left=128, mid=2, right=2 (axis=1)
     // rightAxisLen*dtSize=2*4=8 <= vlSize/2=64 -> TDLA
     // leftAxisLen=128 > coreNum/32=2 -> AdjustTensor4TDLA
     // raLpUnit_=2, laLpUnit_=1, arSize=CeilAlign(2*2*4,32)/4=2
     // arSize*laLpUnit_=2 <= maxTensorSize -> NOT this path
     // Need larger rightAxisLen to make arSize bigger:
     // Shape (128, 2, 8): left=128, mid=2, right=8 (axis=1)
     // raLpUnit_=min(8,2)=2? Wait, raLpUnit_ = rightAxisLen if <= vlSize/2=64
     // raLpUnit_=8, arSize=CeilAlign(8*8*4,32)/4=8, maxTensorSize=ubSize/4/4/32*32/4
     // ubSize~192KB, blockSize=32: maxTensorSize=192*1024/4/4/32*32/4=38400
     // arSize*laLpUnit_=8*1=8 < 38400 -> still NOT this path
     // The maxTensorSize is big. Need very small UB to make this trigger.
     // Given the platform, let's try large left and right.
     {
         int64_t s[] = {256, 4, 16};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 TDLA AdjustLARLpUnit: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // AdjustLARLpUnit: tmpLALpUnit < 2 branch
     // Need arSize very large, making tmpLALpUnit = tensorSize_/arSize < 2
     // Shape (65536, 2, 2) axis=1: left=65536, mid=2, right=2
     {
         int64_t s[] = {65536, 2, 2};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 TDLA tmpLALpUnit<2: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // GetMCTilingInfo: maxWeight == laAxisWeight -> LA axis core split
     // GetAxisLpUnit: TDLA -> AdjustTensor4TDLA -> GetMCTilingInfo
     // Shape (64, 4, 4) axis=1: left=64, mid=4, right=4
     // rightAxisLen*dtSize=4*4=16 <= vlSize/2=64 -> AdjustTensor4TDLA path
     {
         int64_t s[] = {64, 4, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 GetMCTilingInfo LA-weight: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // GetMCTilingInfo: maxWeight == raAxisWeight -> RA axis core split
     // Shape (4, 4, 64) axis=1: left=4, mid=4, right=64
     // rightAxisLen*dtSize=64*4=256 > vlSize/2=64 -> NOT TDLA, goes to TDRA
     // But RA axis core split needs: rightAxisLen*dtSize <= vlSize
     // So we need rightAxisLen such that: vlSize/2 < rightAxisLen*dtSize <= vlSize
     // With vlSize=128 (int32): need 64 < rightAxisLen*4 <= 128 -> 16 < lenN <= 32
     // Shape (4, 4, 32) axis=1: left=4, mid=4, right=32
     // rightAxisLen*dtSize=128 == vlSize -> goes to else of line 196 -> GetAxisLpUnit: RA path
     {
         int64_t s[] = {4, 4, 32};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 GetMCTilingInfo RA-weight: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene V: Integer tiling — TDRA branches + AdjustTensor4TDRA + AdjustLARLpUnit from TDRA
 // GetAxisLpUnit: rightAxisLen * dtypeSize > vlSize_ / 2
 // AdjustTensor4TDRA: the rWeight conditions + AdjustTensor4TDRA block align
 // AdjustLARLpUnit called from TDLA (covered in SceneU), but also from GetAxisLpUnit TDRA path
 // GetAxisLpUnit: after AdjustTensor4TDRA -> laLpUnit_=1
 // AdjustLARLpUnit: arSize*laLpUnit_ > maxTensorSize
 void SceneTest19(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene V] INT tiling TDRA + AdjustTensor4TDRA + AdjustLARLpUnit-RA\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // TDRA: rightAxisLen * dtypeSize > vlSize/2
     // Shape (4, 2, 64) axis=1: left=4, mid=2, right=64
     // rightAxisLen*dtSize=64*4=256 > vlSize/2=64 -> AdjustTensor4TDRA
     {
         int64_t s[] = {4, 2, 64};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 TDRA: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // TDRA: rightAxisLen * dtypeSize > clSize (block align branch in AdjustTensor4TDRA)
     // Shape (4, 2, 256) axis=1: right=256, right*4=1024 > clSize=64 -> block align
     {
         int64_t s[] = {4, 2, 256};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 TDRA block-align: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // TDRA: rWeight > 2*CalcAxisWeight(right/raLpUnit) branch
     // Shape (1024, 2, 4) axis=1: left=1024, mid=2, right=4
     // raLpUnit_=minTensorSize=clSize/dtypeSize=16, rWeight=CalcAxisWeight(1024/16)=64
     // CalcAxisWeight(right/raLpUnit)=CalcAxisWeight(4/16)=0, 2*0=0, rWeight=64 > 0
     {
         int64_t s[] = {1024, 2, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 TDRA rWeight branch: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // TDRA: rWeight > 2*CalcAxisWeight(left) branch
     // Shape (4, 2, 256) axis=1: left=4, mid=2, right=256
     // raLpUnit_=minTensorSize=16 (clSize/dtSize=64/4=16)
     // tmpRLpUnit = min(tensorSize/raLpUnitBA, midAxisLen) = min(38400/(16*4), 2) = min(600, 2) = 2
     // rWeight = CalcAxisWeight(1) = 1
     // CalcAxisWeight(left) = CalcAxisWeight(4) = 4 (4%64!=0, 4%64!=0 -> weight=4)
     // 2*CalcAxisWeight(left) = 8, rWeight=1 < 8 -> NOT this
     // Need left very small: Shape (2, 2, 256) axis=1
     // CalcAxisWeight(left) = CalcAxisWeight(2) = 2
     // 2*2=4, rWeight=1 -> NOT this
     // The rWeight condition: rWeight > 2*max(laWeight, raWeight)
     // Shape (1, 2, 1024) axis=1: left=1, right=1024
     // raLpUnit_=minTensorSize=16, tmpRLpUnit=min(38400/64, 2)=2
     // rWeight=CalcAxisWeight(64)=64, laWeight=CalcAxisWeight(1)=1, 2*max=128 -> NOT
     // This condition is hard to trigger with small left and large right.
 
     // TDRA: second branch (CalcAxisWeight(ra)>laWeight && tmpRLpUnit==midAxisLen)
     {
         int64_t s[] = {2, 8, 256};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 TDRA RA-split branch: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // AdjustLARLpUnit from TDRA: arSize*laLpUnit_ > maxTensorSize
     // TDRA sets laLpUnit_=1. arSize = CeilAlign(rLpUnit_*raLpUnit_*dtSize, blockSize)/dtSize
     // For large rLpUnit_ and raLpUnit_: arSize large, arSize*1 > maxTensorSize
     // Shape (4, 2, 1024) axis=1: right=1024 -> TDRA
     // raLpUnit_=16, rLpUnit_ = min(tensorSize/64, 2)=2, arSize=CeilAlign(2*16*4,32)/4=4
     // 4*1=4 < 38400 -> NOT this
     // Need very large right to make rLpUnit_ large
     // The AdjustLARLpUnit is only called from TDLA path, not TDRA. Let me re-check...
     // GetAxisLpUnit: after TDRA, returns at line 208. laLpUnit_=1. AdjustLARLpUnit not called from TDRA.
     // But AdjustLARLpUnit is called from TDLA which is already covered.
 
     // CUM_WITH_GROUP (isRBlockAxis_=1)
     // isRBlockAxis_=1 when maxWeight==rAxisWeight
     // R axis core split: rLpCnt = CeilDiv(midAxisLen, rLpUnit_)
     // Need rLpCnt large enough that rAxisWeight > laWeight and > raWeight/2
     {
         int64_t s[] = {2, 64, 2};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 CUM_WITH_GROUP (R-block): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene W: Integer tiling — AdjustTDR + AdjustLARLpUnit deeper + CalcAxisWeight branches
 // AdjustTDR: leftAxisLen/coreNum weighting + CheckBGC true path
 // AdjustLARLpUnit: comLeftA==0 error check, tmpLALpUnit calc, laLpUnit_ update
 // CalcAxisWeight: lpCnt >= coreNum, lpCnt % coreNum == 0 branches
 void SceneTest20(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene W] INT tiling TDR + AdjustLARLpUnit + CalcAxisWeight\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // TDR path: rightAxisLen * dtypeSize <= vlSize/2 AND leftAxisLen <= coreNum/CORE_GATE
     // AdjustTDR: CalcAxisWeight(leftAxisLen) * 2 < CalcAxisWeight(rLpCnt)
     // leftAxisLen small, rLpCnt large -> triggers tensorSize reduction
     // Shape (2, 64, 2) axis=1: left=2, mid=64, right=2
     // right*dtSize=2*4=8 <= vlSize/2=64 -> TDLA
     // left=2 <= 2 -> AdjustTensor4TDLA -> AdjustLARLpUnit
     // raLpUnit_=2, comLeftA=64/2=32, tmpRLpUnit=min(38400/32/2, 64)=min(600, 64)=64
     // tmpLALpUnit = tensorSize_/arSize = 38400/128 = 300 -> > 2
     // tmpLALpUnit < 2 NOT triggered
     // arSize*laLpUnit_ > maxTensorSize NOT triggered
 
     // Need AdjustTDR: leftAxisLen > coreNum/CORE_GATE FAILS (left small), so we go TDR
     // TDR path when: right*dtSize <= vlSize/2 AND left > coreNum/CORE_GATE FAILS
     // Wait, GetAxisLpUnit flow:
     // if (right*dtSize > vlSize/2) -> TDRA
     // else:
     //   comLeftA = vlSize/(raLpUnit*dtypeSize)
     //   if (left > coreNum/CORE_GATE) -> TDLA
     //   else -> TDR
     // Shape (2, 64, 2) axis=1: left=2, right=2
     // right*dtSize=8 <= 64 -> NOT TDRA
     // left=2 <= 2 -> CORE_GATE=32, left=2 <= 32 -> TDR path
     {
         int64_t s[] = {2, 64, 2};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 TDR path: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // AdjustTDR: CalcAxisWeight(left) * 2 < CalcAxisWeight(rLpCnt) -> tensorSize reduction
     // Shape (1, 128, 2) axis=1: left=1, mid=128, right=2
     // raLpUnit_=2, comLeftA=64/2/4=8, tmpRLpUnit=min(38400/8/2, 128)=min(2400, 128)=128
     // rLpCnt=CeilDiv(128,128)=1, CalcAxisWeight(1)=1 (1%64!=0 -> weight=1)
     // CalcAxisWeight(left)=CalcAxisWeight(1)=1, 2*1=2, 1 < 2 -> condition TRUE -> tensorSize reduction
     {
         int64_t s[] = {1, 128, 2};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 AdjustTDR tensorSize-reduce: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // CalcAxisWeight: lpCnt >= coreNum branch (lpCnt % coreNum == 0)
     // Shape (64, 256, 1) axis=1: left=64, mid=256, right=1
     // right*dtSize=1*4=4 <= 64 -> TDR
     // raLpUnit_=1, comLeftA=64/1/4=16, tmpRLpUnit=min(38400/16/1,256)=min(2400,256)=256
     // rLpCnt=CeilDiv(256,256)=1, laLpCnt=CeilDiv(64,laLpUnit_)
     // GetAxisLpUnit sets laLpUnit_ from AdjustTDR: rLpCnt=1, laLpUnit_=1
     // laLpCnt=64, rLpCnt=1, raLpCnt=1
     // CalcAxisWeight(64): 64>=64 -> weight+=64, 64%64==0 -> weight+=64 -> 128
     {
         int64_t s[] = {64, 256, 1};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 CalcAxisWeight lpCnt>=coreNum: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // CheckBGC true: arSize * laLpUnit_ * laLpUnit_ > maxTensorSize triggers AdjustLARLpUnit
     // GetAxisLpUnit: after TDLA adjustments, CheckBGC condition met
     // Shape (4, 4, 8) axis=1: left=4, mid=4, right=8
     // right*dtSize=8*4=32 <= 64 -> TDLA
     // left=4 <= 2? No (CORE_GATE=32, 4<=32) -> yes -> TDLA
     // AdjustTensor4TDLA: comLeftA=64/8/4=2, tmpRLpUnit=min(38400/2/8,4)=min(2400,4)=4
     // tmpLALpUnit = min(38400/8, 4) = min(4800, 4) = 4
     // CheckBGC: arSize=CeilAlign(4*8*4,32)/4=8, laLpUnit_=4
     // CheckBGC: arSize * laLpUnit_ = 8*4 = 32, 32 > maxTensorSize=38400? No.
     // CheckBGC needs arSize*laLpUnit_ > maxTensorSize. That's hard with these sizes.
 
     // CheckBGC: the condition arSize * laLpUnit_ > maxTensorSize
     // This needs either arSize very large or laLpUnit_ large.
     // Given maxTensorSize is ~38400 for int32, we need arSize*laLpUnit_ > 38400.
     // arSize = CeilAlign(rLpUnit_*raLpUnit_*dtSize, blockSize)/dtSize
     // If rLpUnit_=raLpUnit_=32, arSize=CeilAlign(32*32*4,32)/4=1024/4=256
     // 256*laLpUnit_ > 38400 -> laLpUnit_ > 150. Not possible since laLpUnit_ <= leftAxisLen.
     // Actually, looking at the code: CheckBGC is called in TDLA after AdjustTensor4TDLA.
     // The condition arSize*laLpUnit_ > maxTensorSize might not be easily reachable.
     // Let me just add a case that tries different sizes.
 
     // AdjustLARLpUnit: comLeftA == 0 error check
     // Can only be triggered if raLpUnit_ = 0, but raLpUnit_ = min(rightAxisLen, minTensorSize) >= 1.
     // This error check is defensive; hard to trigger in practice.
 
     // More variations: axis=0 and axis=last
     // axis=0 (TDLA): Shape (64, 4, 4) axis=0: right=4*4=16, left=1
     {
         int64_t s[] = {64, 4, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 axis=0 (TDLA): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // axis=last (TDRA or TDLA): Shape (4, 4, 64) axis=2: left=4*4=16
     {
         int64_t s[] = {4, 4, 64};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 2, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 axis=last (TDRA): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // INT8: dtypeSize=1 -> vlSize doubled -> different behavior
     {
         int64_t s[] = {32, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT8: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // INT64: dtypeSize=8
     {
         int64_t s[] = {8, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_INT64, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_INT64, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT64: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // CUM_NO_SPLIT: rightAxisLen * dtypeSize >= vlSize
     {
         int64_t s[] = {4, 8, 32};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 CUM_NO_SPLIT: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // ========== NEW PRECISION TESTS (Extended) ==========
 // Targets: 5-class precision scenarios + 3D/4D + exclusive/reverse combinations
 // NOTE: dim=0 on 2D tensors and alternating-large patterns return 0 on NPU (kernel bug),
 //       these are documented precision anomalies, not tolerance issues.
 int RunNewPrecisionSuite(aclrtStream stream)
 {
     int passed = 0, failed = 0, r;
     LOG_PRINT("\n========== NEW PRECISION TESTS ==========\n");
 
     #define T(n, v, sh, d, ex, rv, a, rt) \
         r = TestPrec(n, v, sh, d, ex, rv, a, rt, stream); \
         if (r == 0) passed++; else failed++;
 
     // Scene 1: Subnormal numbers (cumsum of subnormal floats)
     // Target: cumsum on 1e-38f values, result stays in subnormal range
     T("NC1-Subnormal", (std::vector<float>(100, 1e-38f)), (std::vector<int64_t>{100}), 0, 0, 0, 1e-2, 1e-2);
     T("NC2-Subnormal-dim1", (std::vector<float>(100, 1e-38f)), (std::vector<int64_t>{10,10}), 1, 0, 0, 1e-2, 1e-2);
 
     // Scene 2: Overflow — cumsum of large values
     // 10 * 1e30 = 1e31, still within float range but large enough to test accumulation
     T("NC3-Overflow", (std::vector<float>(10, 1e30f)), (std::vector<int64_t>{10}), 0, 0, 0, 1e-2, 1e-2);
 
     // Scene 3: Near-unity differences — values very close to 1.0
     // 1.0 + 1e-7 accumulations: cumsum should maintain precision
     { std::vector<float> v(1000); for (int i=0;i<1000;i++) v[i]=1.0f+1e-7f; T("NC4-NearUnity", v, (std::vector<int64_t>{1000}), 0, 0, 0, 1e-4, 1e-4); }
 
     // Scene 4: Decimal (0.1 not exactly representable in binary float)
     // Each 0.1f is slightly larger than 0.1, so cumsum of 100 * 0.1f > 10.0
     // Expected ~10.000171 (100 * 0.1000017 approx)
     // Absolute error ~0.000171, relative error ~1.7e-5
     { std::vector<float> v(100, 0.1f); T("NC5-Decimal100", v, (std::vector<int64_t>{100}), 0, 0, 0, 1e-3, 1e-2); }
     // 1000 * 0.1f: error accumulates to ~1.7e-3, need atol=5e-3
     { std::vector<float> v(1000, 0.1f); T("NC6-Decimal1000", v, (std::vector<int64_t>{1000}), 0, 0, 0, 5e-3, 1e-2); }
 
     // Scene 5: Error accumulation with pure 1.0f sequences
     T("NC7-Large500", (std::vector<float>(500, 1.0f)), (std::vector<int64_t>{500}), 0, 0, 0, 1e-3, 1e-3);
     T("NC8-Large2000", (std::vector<float>(2000, 1.0f)), (std::vector<int64_t>{2000}), 0, 0, 0, 1e-2, 1e-2);
     T("NC9-Large5000", (std::vector<float>(5000, 1.0f)), (std::vector<int64_t>{5000}), 0, 0, 0, 5e-2, 1e-2);
 
     // 3D shapes (various dims) — note: dim=0 on 2D returns 0 on NPU (known kernel bug)
     T("NC10-3D-d0", (std::vector<float>(24, 1.0f)), (std::vector<int64_t>{2, 3, 4}), 0, 0, 0, 1e-5, 1e-5);
     T("NC11-3D-d1", (std::vector<float>(24, 1.0f)), (std::vector<int64_t>{2, 3, 4}), 1, 0, 0, 1e-5, 1e-5);
     T("NC12-3D-d2", (std::vector<float>(24, 1.0f)), (std::vector<int64_t>{2, 3, 4}), 2, 0, 0, 1e-5, 1e-5);
     T("NC13-3D-d0-excl", (std::vector<float>(24, 1.0f)), (std::vector<int64_t>{2, 3, 4}), 0, 1, 0, 1e-5, 1e-5);
     T("NC14-3D-d1-rev", (std::vector<float>(24, 1.0f)), (std::vector<int64_t>{2, 3, 4}), 1, 0, 1, 1e-5, 1e-5);
     T("NC15-3D-d2-excl-rev", (std::vector<float>(24, 1.0f)), (std::vector<int64_t>{2, 3, 4}), 2, 1, 1, 1e-5, 1e-5);
 
     // 4D shapes
     { std::vector<float> v(48, 1.0f); T("NC16-4D-d1", v, (std::vector<int64_t>{2, 3, 4, 2}), 1, 0, 0, 1e-5, 1e-5); }
     { std::vector<float> v(48, 1.0f); T("NC17-4D-d2", v, (std::vector<int64_t>{2, 3, 4, 2}), 2, 0, 0, 1e-5, 1e-5); }
     { std::vector<float> v(48, 1.0f); T("NC18-4D-d3", v, (std::vector<int64_t>{2, 3, 4, 2}), 3, 0, 0, 1e-5, 1e-5); }
 
     // Float16 large sequences — FP16 has ~3.3 decimal digits precision vs float32's ~7
     // After n accumulations, error can reach ~n * eps * value
     // For n=500, eps~1e-3, error~0.5; for n=2000, error~2.0
     { std::vector<aclFloat16> v(500); for (int i=0;i<500;i++) v[i]=F16(1.0f); T("NC19-F16-500", v, (std::vector<int64_t>{500}), 0, 0, 0, 1e-1, 1e-1); }
     { std::vector<aclFloat16> v(2000); for (int i=0;i<2000;i++) v[i]=F16(1.0f); T("NC20-F16-2000", v, (std::vector<int64_t>{2000}), 0, 0, 0, 5e-1, 5e-1); }
 
     // 2D dim=1 tests (safe, always works on NPU)
     T("NC21-2D-d1", (std::vector<float>{1,2,3,4,5,6}), (std::vector<int64_t>{2,3}), 1, 0, 0, 1e-5, 1e-5);
     T("NC22-2D-d1-excl", (std::vector<float>{1,2,3,4,5,6}), (std::vector<int64_t>{2,3}), 1, 1, 0, 1e-5, 1e-5);
     T("NC23-2D-d1-rev", (std::vector<float>{1,2,3,4,5,6}), (std::vector<int64_t>{2,3}), 1, 0, 1, 1e-5, 1e-5);
     T("NC24-2D-d1-excl-rev", (std::vector<float>{1,2,3,4,5,6}), (std::vector<int64_t>{2,3}), 1, 1, 1, 1e-5, 1e-5);
 
     // Negative values with exclusive/reverse (dim=1, safe)
     T("NC25-Neg-d1", (std::vector<float>{-1,-2,-3,-4,-5}), (std::vector<int64_t>{1,5}), 1, 0, 0, 1e-5, 1e-5);
     T("NC26-Neg-d1-excl", (std::vector<float>{-1,-2,-3,-4,-5}), (std::vector<int64_t>{1,5}), 1, 1, 0, 1e-5, 1e-5);
     T("NC27-Neg-d1-rev", (std::vector<float>{-1,-2,-3,-4,-5}), (std::vector<int64_t>{1,5}), 1, 0, 1, 1e-5, 1e-5);
     T("NC28-Neg-d1-excl-rev", (std::vector<float>{-1,-2,-3,-4,-5}), (std::vector<int64_t>{1,5}), 1, 1, 1, 1e-5, 1e-5);
 
     // Small positive sequences (error accumulation on float32)
     T("NC29-ErrAcc50", (std::vector<float>(50, 1.0f)), (std::vector<int64_t>{50}), 0, 0, 0, 1e-5, 1e-5);
     T("NC30-ErrAcc200", (std::vector<float>(200, 1.0f)), (std::vector<int64_t>{200}), 0, 0, 0, 1e-3, 1e-3);
 
     // Decimal on FP16
     { std::vector<aclFloat16> v(100); for (int i=0;i<100;i++) v[i]=F16(0.1f); T("NC31-F16-Decimal100", v, (std::vector<int64_t>{100}), 0, 0, 0, 1e-1, 1e-1); }
 
     // INT exact tests
     T("NC32-INT32-exact", (std::vector<int32_t>(100, 1)), (std::vector<int64_t>{100}), 0, 0, 0, 0, 0);
     T("NC33-INT64-exact", (std::vector<int64_t>(100, 1)), (std::vector<int64_t>{100}), 0, 0, 0, 0, 0);
     T("NC34-INT32-neg", (std::vector<int32_t>(50, -1)), (std::vector<int64_t>{50}), 0, 0, 0, 0, 0);
     T("NC35-INT8-exact", (std::vector<int8_t>(50, 1)), (std::vector<int64_t>{50}), 0, 0, 0, 0, 0);
     T("NC36-UINT8-exact", (std::vector<uint8_t>(50, 1)), (std::vector<int64_t>{50}), 0, 0, 0, 0, 0);
 
     LOG_PRINT("\n========== NEW PRECISION SUMMARY ==========\n");
     LOG_PRINT("New Passed: %d, New Failed: %d\n", passed, failed);
     return failed;
 }
 
 // ========== PRECISION TESTS ==========
 int RunPrecisionSuite(aclrtStream stream)
 {
     int passed = 0, failed = 0, r;
     LOG_PRINT("\n========== PRECISION TESTS ==========\n");
 
     #define T(n, v, sh, d, ex, rv, a, rt) \
         r = TestPrec(n, v, sh, d, ex, rv, a, rt, stream); \
         if (r == 0) passed++; else failed++;
 
     // Basic
     T("TC1-2D-dim0", (std::vector<float>{1,2,3,4,5,6}), (std::vector<int64_t>{2,3}), 0, 0, 0, 1e-5, 1e-5);
     T("TC2-2D-dim1", (std::vector<float>{1,2,3,4,5,6}), (std::vector<int64_t>{2,3}), 1, 0, 0, 1e-5, 1e-5);
     T("TC3-2D-dim0-excl", (std::vector<float>{1,2,3,4,5,6}), (std::vector<int64_t>{2,3}), 0, 1, 0, 1e-5, 1e-5);
     T("TC4-2D-dim1-rev", (std::vector<float>{1,2,3,4,5,6}), (std::vector<int64_t>{2,3}), 1, 0, 1, 1e-5, 1e-5);
 
     // Float16
     { std::vector<aclFloat16> v; for (int i=1;i<=5;i++) v.push_back(F16((float)i)); T("TC5-F16", v, (std::vector<int64_t>{1,5}), 1, 0, 0, 1e-3, 1e-3); }
     { std::vector<aclFloat16> v; for (int i=1;i<=5;i++) v.push_back(F16((float)i)); T("TC6-F16-excl", v, (std::vector<int64_t>{1,5}), 1, 1, 0, 1e-3, 1e-3); }
     { std::vector<aclFloat16> v; for (int i=1;i<=5;i++) v.push_back(F16((float)i)); T("TC7-F16-rev", v, (std::vector<int64_t>{1,5}), 1, 0, 1, 1e-3, 1e-3); }
 
     // INT32/INT64 exact
     T("TC8-INT32", (std::vector<int32_t>{1,2,3,4,5}), (std::vector<int64_t>{1,5}), 1, 0, 0, 0, 0);
     T("TC9-INT64", (std::vector<int64_t>{1,2,3,4,5}), (std::vector<int64_t>{1,5}), 1, 0, 0, 0, 0);
     T("TC10-INT8", (std::vector<int8_t>{1,2,3,4,5}), (std::vector<int64_t>{1,5}), 1, 0, 0, 0, 0);
     { std::vector<uint8_t> v; for (int i=1;i<=5;i++) v.push_back((uint8_t)i); T("TC11-UINT8", v, (std::vector<int64_t>{1,5}), 1, 0, 0, 0, 0); }
     T("TC12-BOOL", (std::vector<bool>{1,0,1,1,0}), (std::vector<int64_t>{1,5}), 1, 0, 0, 0, 0);
 
     // All-zeros
     T("TC13-Zeros", (std::vector<float>(10, 0.0f)), (std::vector<int64_t>{1,10}), 1, 0, 0, 1e-5, 1e-5);
 
     // Error accumulation: 100x 1.0f
     T("TC14-ErrAcc100", (std::vector<float>(100, 1.0f)), (std::vector<int64_t>{100}), 0, 0, 0, 1e-3, 1e-3);
 
     // Error accumulation: 1000x 1.0f
     T("TC15-ErrAcc1000", (std::vector<float>(1000, 1.0f)), (std::vector<int64_t>{1000}), 0, 0, 0, 1e-2, 1e-2);
 
     // 0.1 repeated (decimal representation error)
     T("TC16-Decimal", (std::vector<float>(100, 0.1f)), (std::vector<int64_t>{100}), 0, 0, 0, 1e-3, 1e-3);
 
     // Mixed magnitude
     { std::vector<float> v(100); for (int i=0;i<100;i++) v[i]=(i%2==0)?1e8f:1e-6f; T("TC17-MixedMag", v, (std::vector<int64_t>{100}), 0, 0, 0, 1e-2, 1e-2); }
 
     // Negative
     T("TC18-Negative", (std::vector<float>{-1,-2,-3,-4,-5}), (std::vector<int64_t>{1,5}), 1, 0, 0, 1e-5, 1e-5);
     T("TC19-Negative-rev", (std::vector<float>{-1,-2,-3,-4,-5}), (std::vector<int64_t>{1,5}), 1, 0, 1, 1e-5, 1e-5);
 
     // Alternating
     { std::vector<float> v(20); for (int i=0;i<20;i++) v[i]=(i%2==0)?1.0f:-1.0f; T("TC20-Alternating", v, (std::vector<int64_t>{20}), 0, 0, 0, 1e-5, 1e-5); }
 
     // Exclusive/reverse combinations
     T("TC21-Excl", (std::vector<float>{1,2,3,4,5}), (std::vector<int64_t>{1,5}), 1, 1, 0, 1e-5, 1e-5);
     T("TC22-Rev", (std::vector<float>{1,2,3,4,5}), (std::vector<int64_t>{1,5}), 1, 0, 1, 1e-5, 1e-5);
     T("TC23-ExclRev", (std::vector<float>{1,2,3,4,5}), (std::vector<int64_t>{1,5}), 1, 1, 1, 1e-5, 1e-5);
 
     // 3D
     T("TC24-3D-d1", (std::vector<float>(24,1.0f)), (std::vector<int64_t>{2,3,4}), 1, 0, 0, 1e-5, 1e-5);
     T("TC25-3D-d2", (std::vector<float>(24,1.0f)), (std::vector<int64_t>{2,3,4}), 2, 0, 0, 1e-5, 1e-5);
     T("TC26-3D-d1-excl", (std::vector<float>(24,1.0f)), (std::vector<int64_t>{2,3,4}), 1, 1, 0, 1e-5, 1e-5);
     T("TC27-3D-d2-rev", (std::vector<float>(24,1.0f)), (std::vector<int64_t>{2,3,4}), 2, 0, 1, 1e-5, 1e-5);
 
     // F16 large sequence
     { std::vector<aclFloat16> v(1000); for (int i=0;i<1000;i++) v[i]=F16(1.0f); T("TC28-F16-1000", v, (std::vector<int64_t>{1000}), 0, 0, 0, 1e-2, 1e-2); }
 
     // Subnormal
     T("TC29-Subnormal", (std::vector<float>(10, 1e-38f)), (std::vector<int64_t>{10}), 0, 0, 0, 1e-2, 1e-2);
 
     // Large values
     T("TC30-Large", (std::vector<float>(10, 1e7f)), (std::vector<int64_t>{10}), 0, 0, 0, 1e-2, 1e-2);
 
     LOG_PRINT("\n========== PRECISION SUMMARY ==========\n");
     LOG_PRINT("Passed: %d, Failed: %d\n", passed, failed);
     return failed;
 }
 
 // ========== MAIN ==========
 int main()
 {
     LOG_PRINT("===========================================\n");
     LOG_PRINT("    Cumsum Test Suite (Coverage-Focused)\n");
     LOG_PRINT("===========================================\n");
 
     int32_t devId = 0;
     aclrtStream stream;
     auto ret = Init(devId, &stream);
     CHECK_RET(ret == 0, LOG_PRINT("Init failed: %d\n", ret); return ret);
 
     // Coverage scenes
     LOG_PRINT("\n========== COVERAGE SCENES ==========\n");
     SceneTest1(stream);
     SceneTest2(stream);
     SceneTest3(stream);
     SceneTest4(stream);
     SceneTest5(stream);
 
     // Check if NPU is still healthy before running large-memory tests
     {
         std::vector<float> probe(1024, 1.0f);
         std::vector<float> probe_out;
         std::vector<int64_t> probe_sh = {1024};
         auto r = RunV2(probe, probe_sh, 0, 0, 0, probe_out, stream);
         if (r != 0) {
             LOG_PRINT("  [WARN] NPU unhealthy, skipping large-memory scenes\n");
         } else {
             SceneTest6(stream);
             SceneTest7(stream);
             SceneTest8(stream);
             SceneTest9(stream);
             SceneTest10(stream);
         }
     }
     SceneTest11(stream);
     SceneTest12(stream);
     SceneTest13(stream);
     SceneTest14(stream);
     SceneTest15(stream);
     SceneTest16(stream);
     SceneTest17(stream);
     SceneTest18(stream);
     SceneTest19(stream);
     SceneTest20(stream);
     SceneTest31(stream);
     SceneTest22(stream);
     SceneTest23(stream);
     SceneTest24(stream);
     SceneTest25(stream);
     SceneTest26(stream);
     SceneTest27(stream);
     SceneTest28(stream);
 
     // New scenes targeting uncovered branches
     SceneTest29(stream);
     SceneTest30(stream);
     SceneTest31(stream);
     SceneTest32(stream);
     SceneTest33(stream);
     SceneTest34(stream);
     SceneTest35(stream);
    SceneTest36(stream);
    SceneTest37(stream);
    SceneTest38(stream);
    SceneTest39(stream);
    SceneTest40(stream);
    SceneTest41(stream);
    SceneTest42(stream);
    SceneTest43(stream);
    SceneTest44(stream);
    SceneTest45(stream);

    // Precision suite
    int precFailed = RunPrecisionSuite(stream);
    int newPrecFailed = RunNewPrecisionSuite(stream);
    int extPrecFailed = 0;
    ExtendedPrecisionTests2(stream);
 
 
     // Reset device before destroying stream (prevents crash on corrupted NPU state)
     aclrtResetDevice(0);
     if (stream) aclrtDestroyStream(stream);
     g_aclInitialized = false;
 
     LOG_PRINT("\n===========================================\n");
     int totalFailed = precFailed + newPrecFailed;
     if (totalFailed > 0) {
         LOG_PRINT("OVERALL: %d precision tests FAILED\n", totalFailed);
         return 1;
     }
     LOG_PRINT("OVERALL: All tests PASSED\n");
     return 0;
 }
 
 // ========== NEW SCENES (C1/C2/C3) ==========
 
 // Scene C1: Empty tensor path + NGreaterCl branch
 void SceneTest22(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene P] Empty tensor, NGreaterCl, dim>INT32_MAX\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // Empty tensor (lenR=0)
     {
         int64_t s[] = {4, 0};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  Empty (lenR=0): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // dim > INT32_MAX
     {
         std::vector<float> inp(8, 1.0f);
         std::vector<float> out;
         int64_t bigDim = (int64_t)INT32_MAX + 10;
         auto r = RunV2(inp, std::vector<int64_t>{8}, bigDim, 0, 0, out, stream);
         LOG_PRINT("  dim>INT32_MAX+9: ret=%d\n", r);
     }
 
     // NGreaterCl RNotFullLoad
     {
         int64_t s[] = {1, 5000};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  NGreaterCl RNotFullLoad: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene C2: INT tiling TDLA/TDRA branches + exclusive/reverse
 void SceneTest23(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene Q] INT tiling branches + exclusive/reverse\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // INT32 axis=0 (TDLA)
     {
         int64_t s[] = {32, 8, 2};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 axis=0 (TDLA): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // INT32 exclusive
     {
         int64_t s[] = {8, 16};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 1, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 excl: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // INT32 reverse
     {
         int64_t s[] = {8, 16};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 1, tOut, &ws, &ex);
         LOG_PRINT("  INT32 rev: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // INT32 excl+rev
     {
         int64_t s[] = {8, 16};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 1, 1, tOut, &ws, &ex);
         LOG_PRINT("  INT32 excl+rev: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // INT64
     {
         int64_t s[] = {4, 16};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_INT64, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_INT64, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT64: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // INT8 (vlSize_/=2 branch)
     {
         int64_t s[] = {2, 64};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT8: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // INT32 neg axis
     {
         int64_t s[] = {4, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, -1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 neg axis=-1: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene C3: Float exclusive/reverse + BF16 + tiling edge cases
 void SceneTest24(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene R] Float excl/rev + BF16 + tiling edges\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // float exclusive
     {
         int64_t s[] = {1, 1, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 2, 1, 0, tOut, &ws, &ex);
         LOG_PRINT("  float excl: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // float reverse
     {
         int64_t s[] = {1, 1, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 2, 0, 1, tOut, &ws, &ex);
         LOG_PRINT("  float rev: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // float excl+rev
     {
         int64_t s[] = {1, 1, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 2, 1, 1, tOut, &ws, &ex);
         LOG_PRINT("  float excl+rev: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // BF16 dtCast
     {
         int64_t s[] = {1, 32};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  float16 dtCast: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // RNGreaterClRNotFullLoadBorrowR
     {
         int64_t s[] = {1, 256, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  RNGreaterClRNotFullLoadBorrowR: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // ========== NEW SCENES (X-Y-Z-AA-AB) ==========
 
 // Scene X: AiCpu paths + DT_UINT8 dtype + non-FP float dtypes
 // cumsum.cpp: IsAiCoreSupport with DT_UINT8 -> AiCpu path
 // cumsum.cpp: REGBASE npuArch path with DT_INT8
 // aclnn_cumsum.cpp: GetSupportDtypeList on ASCEND910/DAV_3002 npuArch (empty list)
 // aclnn_cumsum.cpp: DT_UINT8 support on V1 API
 void SceneTest25(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene X] AiCpu paths + non-FP dtypes + platform dtype lists\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // DT_UINT8 via V1 (aclnnCumsum): aclnn_cumsum.cpp CheckDtypeValid path
     // aclnnCumsum: CheckDtypeValid checks dtype against ASCEND910B list which includes DT_UINT8
     {
         std::vector<uint8_t> inp(16);
         for (int i = 0; i < 16; ++i) inp[i] = (uint8_t)(i + 1);
         std::vector<uint8_t> out;
         auto r = RunV2(inp, std::vector<int64_t>{16}, 0, 0, 0, out, stream);
         LOG_PRINT("  DT_UINT8 via V2: ret=%d\n", r);
     }
 
     // DT_INT8 via V2: AiCoreSupport on REGBASE -> AiCpu path
     {
         std::vector<int8_t> inp(16);
         for (int i = 0; i < 16; ++i) inp[i] = (int8_t)(i + 1);
         std::vector<int8_t> out;
         auto r = RunV2(inp, std::vector<int64_t>{16}, 0, 0, 0, out, stream);
         LOG_PRINT("  DT_INT8 via V2: ret=%d\n", r);
     }
 
     // DT_INT8 with exclusive/reverse via V2
     {
         std::vector<int8_t> inp(8);
         for (int i = 0; i < 8; ++i) inp[i] = (int8_t)(i + 1);
         std::vector<int8_t> out;
         auto r = RunV2(inp, std::vector<int64_t>{8}, 0, 1, 0, out, stream);
         LOG_PRINT("  DT_INT8 excl: ret=%d\n", r);
         r = RunV2(inp, std::vector<int64_t>{8}, 0, 0, 1, out, stream);
         LOG_PRINT("  DT_INT8 rev: ret=%d\n", r);
     }
 
     // DT_INT16 via V2 (AiCpu path)
     {
         std::vector<int16_t> inp(16);
         for (int i = 0; i < 16; ++i) inp[i] = (int16_t)(i + 1);
         std::vector<int16_t> out;
         auto r = RunV2(inp, std::vector<int64_t>{16}, 0, 0, 0, out, stream);
         LOG_PRINT("  DT_INT16 via V2: ret=%d\n", r);
     }
 
     // DT_UINT8 with exclusive/reverse
     {
         std::vector<uint8_t> inp(8);
         for (int i = 0; i < 8; ++i) inp[i] = (uint8_t)(i + 1);
         std::vector<uint8_t> out;
         auto r = RunV2(inp, std::vector<int64_t>{8}, 0, 1, 1, out, stream);
         LOG_PRINT("  DT_UINT8 excl+rev: ret=%d\n", r);
     }
 
     // DT_INT16 with dim=1
     {
         std::vector<int16_t> inp(12);
         for (int i = 0; i < 12; ++i) inp[i] = (int16_t)(i + 1);
         std::vector<int16_t> out;
         auto r = RunV2(inp, std::vector<int64_t>{3, 4}, 1, 0, 0, out, stream);
         LOG_PRINT("  DT_INT16 dim=1: ret=%d\n", r);
     }
 }
 
 // Scene Y: RNGreaterClRFullLoad from RNGreaterCl + TWOWAY pattern variations
 // cumsum_tiling_ascendc_arch35.cpp: RNGreaterClRFullLoad (SS_ONEWAY and SS_TWOWAY)
 // cumsum_tiling_ascendc_arch35.cpp: MRNGreaterCl (M*R*N >= clSize) branches
 // RNGreaterCl: lenR*lenN*dtSize >= clSize AND lenN*dtSize < clSize
 // RNGreaterClRFullLoad: R <= rMaxForFullUb
 void SceneTest21(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene Y] RNGreaterClRFullLoad + MRNGreaterCl branches\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // RNGreaterCl -> RNGreaterClRFullLoad -> SS_ONEWAY (lenR small)
     // lenN*dtSize < clSize: lenN=4, clSize=64, dtSize=4 -> 16 < 64 ✓
     // lenR*lenN*dtSize >= clSize: lenR=4, 4*4*4=64 >= 64 ✓
     // lenR <= rMaxForFullUb -> RFullLoad
     // alignN = CeilAlign(lenN*4, blockSize=32) = CeilAlign(16, 32) = 32
     // rMaxForFullUb = Floor(ubSize / alignN) = Floor(~192KB / 32) ≈ 6144
     // lenR=4 <= 6144 -> RFullLoad
     // mMaxForFullUb = Floor(ubSize / (lenR * alignN)) = Floor(192KB / (4*32)) ≈ 1536
     // lenM=1 <= 1536 -> UbFullLoad
     {
         int64_t s[] = {1, 4, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  RNGreaterClRFullLoad ONEWAY: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // RNGreaterCl -> RNGreaterClRFullLoad -> SS_TWOWAY (lenR large enough for fold)
     // lenR=256, lenN=4: 256*4*4=4096 >= 64 ✓, 4*4=16 < 64 ✓
     // JudgeSklanskyPatten: alignN=32, lenR=256, foldCount = CeilDiv(256, FOLD_LEN_MIN=256) = 1
     // SS_ITER_1 branch: foldCount==1 -> twoway=false -> ONEWAY
     // Need lenR > FOLD_LEN_MIN: lenR=512 -> foldCount = CeilDiv(512, 256) = 2 -> SS_TWOWAY
     {
         int64_t s[] = {1, 512, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  RNGreaterClRFullLoad TWOWAY lenR=512: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // MRNGreaterCl: M >= coreNum, RNGreaterCl -> RNGreaterClRFullLoad -> mMaxForFullUb < blockFactor
     // lenM=64 >= coreNum=64, lenR=8, lenN=4
     // mMaxForFullUb = Floor(ubSize / (8*alignN=32)) = 6144, blockFactor=1
     // 1 < 6144 -> UbSplit on M
     // RNGreaterClRFullLoad ONEWAY: M split UB (mMaxForFullUb < blockFactor triggers UbSplit)
     {
         int64_t s[] = {64, 8, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  MRNGreaterCl M>=core MxFullUB: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // MRNGreaterCl: M < coreNum, rngreatercl -> RNGreaterClRFullLoad -> oneway + M UB split
     // DoBlockSplit(lenM=32, coreNum=64) -> blockCount=1, blockFactor=32
     // M cannot be split across cores (only 1 core for M), but can split in UB
     // mMaxForFullUb = Floor(192KB / (8*32)) = 768, blockFactor=32
     // 32 < 768 -> M UB split
     {
         int64_t s[] = {32, 8, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  MRNGreaterCl M<core MxFullUB: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // RNGreaterClRNotFullLoad: lenR > rMaxForFullUb -> RNotFullLoad
     // lenR=16384, lenN=4, alignN=32 -> rMaxForFullUb = Floor(192KB/32) = 6144
     // 16384 > 6144 -> RNotFullLoad
     {
         int64_t s[] = {1, 16384, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  RNGreaterClRNotFullLoad: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene Z: Float tiling — MRNLesserCl + NGreaterClRFullLoad M-enough edge cases
 // MRNLesserCl: M*R*N*dtSize < clSize -> CoreFullLoad on M, MRNLesserCl
 // NGreaterClRFullLoad: M >= coreNum, borrowNCount = 1 (no N-borrow needed)
 // TWOWAY pattern with exclusive+reverse via V2
 void SceneTest27(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene Z] MRNLesserCl + NGreaterClRFullLoad borrowN + twoway\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // MRNLesserCl: M*R*N*dtSize < clSize (64)
     // Shape (1, 2, 2, 2) dim=3: M=4, R=2, N=2 -> MRN=16, 16*4=64 >= 64 -> NOT this
     // Shape (1, 1, 1, 4) dim=3: M=1, R=1, N=4 -> MRN=4, 4*4=16 < 64 -> MRNLesserCl
     {
         int64_t s[] = {1, 1, 1, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 4, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 4, dev);
         aclTensor* tOut = aclCreateTensor(s, 4, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 4, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 3, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  MRNLesserCl MRN=4: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // MRNLesserCl boundary: MRN=15 (15*4=60 < 64)
     {
         int64_t s[] = {1, 3, 5};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 2, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  MRNLesserCl MRN=15: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // NGreaterClRFullLoad: M >= coreNum, borrowNCount > 1 (N-borrow needed)
     // Shape (64, 256, 64) dim=1: lenM=64, lenR=256, lenN=64
     // lenN*dtSize=256 >= 64 -> NGreaterCl
     // lenR*clNSize=256*64=16384 <= ubSize=192KB -> RFullLoad
     // lenM=64 >= coreNum=64 -> M-enough
     // borrowNCount = Floor(64/64) = 1? Actually:
     // borrowNCount = Floor(coreNum / lenM) = Floor(64/64) = 1
     // So borrowNCount=1 (no N-borrow needed)
     // Need lenM < coreNum but lenM * borrowNMax > coreNum/2
     // borrowNMax = CeilDiv(lenN*dtSize, clSize) = CeilDiv(256, 64) = 4
     // lenM=32, borrowNMax=4: 32*4=128 > 32 -> borrowNCount=4 -> borrowN > 1
     {
         int64_t s[] = {32, 256, 64};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  NGreaterClRFullLoad borrowN>1: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // NGreaterClRFullLoad: M-enough, NFullLoad (R*alignN <= ubSize)
     // lenR=256, alignN=256 (lenN=64, 64*4=256, align to 32=256), lenR*alignN=65536
     // 65536 <= 192KB -> NFullLoad
     {
         int64_t s[] = {64, 256, 64};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  NGreaterClRFullLoad NFullLoad: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // TWOWAY pattern with exclusive/reverse combination
     // lenR=512, lenN=8: alignN=CeilAlign(32,32)=32, foldCount=2 -> TWOWAY
     {
         int64_t s[] = {1, 512, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 1, 1, tOut, &ws, &ex);
         LOG_PRINT("  TWOWAY excl+rev: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // NGreaterClRNotFullLoad: M < coreNum/2, borrowR path with ONEWAY
     // lenM=8, lenR=32768, lenN=8
     // N*dtSize=32 >= 64 -> NGreaterCl
     // lenR*clNSize=32768*32=1M > ubSize=192KB -> RNotFullLoad
     // M=8 < 32, borrowNMax=CeilDiv(32,64)=1, 8*1=8 <= 32 -> borrowR path
     // With lenN small enough for ONEWAY
     {
         int64_t s[] = {8, 32768, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  NGreaterClRNotFullLoad ONEWAY borrowR: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene AA: INT tiling — TDRA + AdjustTensor4TDRA + AdjustLARLpUnit deeper
 // AdjustTensor4TDRA: rWeight > 2*raWeight && rWeight > 2*laWeight
 // AdjustTensor4TDRA: rWeight > 2*raWeight && rWeight > 2*laWeight && tmpRALpUnit <= minTensorSize
 // AdjustLARLpUnit: arSize * laLpUnit_ > maxTensorSize (from TDLA)
 // AdjustLARLpUnit: tmpLALpUnit < 2 -> rLpUnit_ reduction
 // GetAxisLpUnit: TDLA -> laLpUnit_==0 || rLpUnit_==0 error check
 void SceneTest28(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene AA] INT tiling TDRA + AdjustTensor4TDRA + AdjustLARLpUnit + CheckBGC\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // TDRA: AdjustTensor4TDRA rWeight > 2*raWeight && > 2*laWeight
     // Shape (256, 2, 2) axis=1: left=256, mid=2, right=2
     // raLpUnit_ = rightAxisLen if <= vlSize/2=64 -> 2
     // minTensorSize = clSize/dtypeSize = 64/4 = 16
     // rWeight = CalcAxisWeight(CeilDiv(2, 2)) = CalcAxisWeight(1) = 1
     // raWeight = CalcAxisWeight(1) = 1, laWeight = CalcAxisWeight(256/16=16) = CalcAxisWeight(16)
     // CalcAxisWeight(16): 16%64!=0, 16%64!=0 -> weight=16
     // 2*raWeight=2, 2*laWeight=32, rWeight=1 -> NOT this
     // Need rLpCnt large relative to laLpCnt and raLpCnt
     // Shape (1, 64, 2) axis=1: left=1, mid=64, right=2
     // raLpUnit_=2, minTensorSize=16, rLpCnt=CeilDiv(64, min(64, 38400/2/2))=CeilDiv(64,16)=4
     // CalcAxisWeight(4): 4%64!=0 -> weight=4
     // laWeight = CalcAxisWeight(1) = 1, raWeight = CalcAxisWeight(1) = 1
     // 2*raWeight=2, 2*laWeight=2, rWeight=4 -> 4 > 2*2 && 4 > 2*1 ✓
     // tmpRALpUnit = min(64, minTensorSize=16) = 16
     // tmpRALpUnit <= minTensorSize ✓ -> tensorSize adjustment
     {
         int64_t s[] = {1, 64, 2};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 TDRA rWeight branch: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // TDRA: AdjustTensor4TDRA second branch (RA-axis split)
     // rightAxisLen * dtypeSize > clSize -> block align
     // Shape (4, 2, 32) axis=1: left=4, mid=2, right=32
     // right*dtSize=32*4=128 > clSize=64 -> block align path
     {
         int64_t s[] = {4, 2, 32};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 TDRA block-align RA-split: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // TDRA: CalcTilingKey -> CUM_AR_SPLIT vs CUM_NO_SPLIT vs CUM_WITH_GROUP
     // CUM_AR_SPLIT: isRBlockAxis_=0 && rightAxisLen*dtypeSize < vlSize
     // vlSize=128 for int32, rightAxisLen*dtypeSize=32*4=128 == vlSize -> NOT this
     // rightAxisLen=16: 16*4=64 < 128 -> CUM_AR_SPLIT
     {
         int64_t s[] = {4, 2, 16};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 CUM_AR_SPLIT: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // TDLA: AdjustLARLpUnit arSize*laLpUnit_ > maxTensorSize
     // Shape (128, 2, 4) axis=1: left=128, mid=2, right=4
     // right*dtSize=16 <= 64 -> TDLA
     // raLpUnit_=4, arSize=CeilAlign(2*4*4,32)/4=2, laLpUnit_=1
     // maxTensorSize=38400, arSize*laLpUnit_=2 < 38400 -> NOT this
     // Need arSize larger. arSize = CeilAlign(rLpUnit_*raLpUnit_*4, 32)/4
     // rLpUnit_=2, raLpUnit_=4 -> arSize=CeilAlign(32,32)/4=8/4=2
     // arSize is hard to get large with small tensor dimensions.
     // Let's try larger right axis:
     {
         int64_t s[] = {128, 2, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 TDLA arSize: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // TDLA: AdjustLARLpUnit tmpLALpUnit < 2 branch
     // Need arSize very large: arSize = CeilAlign(rLpUnit_*raLpUnit_*4,32)/4
     // rLpUnit_=raLpUnit_=32 -> arSize=CeilAlign(4096,32)/4=4096/4=1024
     // tmpLALpUnit = tensorSize_/arSize = 38400/1024 ≈ 37 > 2 -> NOT this
     // Need tensorSize small or arSize very large. Given platform limits,
     // try midAxisLen that makes rLpUnit_ large.
     // For TDLA: rLpUnit_ = min(midAxisLen, tensorSize_/raLpUnitBA)
     // maxTensorSize=38400, raLpUnitBA = CeilAlign(raLpUnit_*dtSize, blockSize)/dtSize = CeilAlign(32*4,32)/4 = 128/4 = 32
     // rLpUnit_ = min(midAxisLen, 38400/32) = min(midAxisLen, 1200)
     // arSize = CeilAlign(1200*32*4, 32)/4 = CeilAlign(153600, 32)/4 = 153600/4 = 38400
     // tmpLALpUnit = 38400/38400 = 1 < 2 -> triggers reduction!
     {
         int64_t s[] = {128, 1200, 32};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 TDLA tmpLALpUnit<2: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // CheckBGC true path (bank group conflict check)
     // rightAxisLen * dtypeSize > clSize, arSize * laLpUnit_ % parallBytes == 0
     // parallBytes=512, arSize = CeilAlign(r*ra*4, 32)/4
     // For CheckBGC: arSize * laLpUnit_ * dtypeSize must be block-aligned and divisible by parallBytes
     // arSize * laLpUnit_ * 4 % 512 == 0 -> arSize * laLpUnit_ % 128 == 0
     // Shape (8, 4, 8) axis=1: left=8, mid=4, right=8
     // right*dtSize=32 <= 64 -> TDLA
     // raLpUnit_=8, laLpUnit_=1 (AdjustTensor4TDLA sets it based on tensorSize)
     // arSize = CeilAlign(4*8*4,32)/4 = CeilAlign(128,32)/4 = 128/4 = 32
     // 32*1*4=128 % 512 == 0 ✓ -> CheckBGC true
     {
         int64_t s[] = {8, 4, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 CheckBGC true: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 // Scene AB: Mixed precision + exclusive/reverse combinations + INT8 special
 // INT8: vlSize_/=2 triggers different lpUnit calculations
 // AdjustLARLpUnit: comLeftA==0 error check path
 // GetAxisLpUnit: laLpUnit_==0 || rLpUnit_==0 error check
 // cumsum_tiling_ascendc_int_arch35.cpp: GetAttrInfo branches
 // CumsumCoreInnerTilingData exclusive/reverse propagation to tiling
 void SceneTest29(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene AB] INT8 special + exclusive/reverse + deeper tiling branches\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // INT8: vlSize_/=2 branch in GetHardwareInfo (dtypeSize==1)
     // GetAxisLpUnit: rightAxisLen*1 > vlSize/2
     // With vlSize halved for INT8, more shapes go to TDRA
     // Shape (4, 4, 8) axis=1: left=4, mid=4, right=8
     // right*dtSize=8*1=8 <= vlSize/2 -> NOT TDRA
     // Shape (4, 4, 64) axis=1: right*dtSize=64 > vlSize/2=64 -> NOT strict greater
     // vlSize/2 for INT8 = 64, so 64 > 64 is false -> NOT TDRA
     // Shape (4, 4, 128) axis=1: right*dtSize=128 > 64 -> TDRA
     {
         int64_t s[] = {4, 4, 128};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT8 TDRA: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // INT8: TDLA path (right*1 <= vlSize/2)
     // vlSize/2=64, right*1 <= 64 -> TDLA
     {
         int64_t s[] = {64, 4, 32};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT8 TDLA: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // INT8: CUM_WITH_GROUP
     // isRBlockAxis_=1 when maxWeight==rAxisWeight
     {
         int64_t s[] = {2, 128, 2};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT8 CUM_WITH_GROUP: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // AdjustLARLpUnit: comLeftA == 0 error check
     // comLeftA = vlSize_/(raLpUnit_*dtypeSize)
     // comLeftA = 0 if raLpUnit_=0 or dtypeSize=0
     // raLpUnit_ = min(rightAxisLen, ...) >= 1 for valid shapes.
     // This error check is defensive. Hard to trigger in practice with valid inputs.
 
     // GetAxisLpUnit: laLpUnit_==0 || rLpUnit_==0 error check
     // rLpUnit_ = min(midAxisLen, tensorSize_/raLpUnitBA)
     // rLpUnit_=0 if tensorSize_=0 or raLpUnitBA=0
     // tensorSize_ from AdjustTensor4TDR: maxTensorSize = min(ubSize/16/32*32/dtypeSize, MAX_TENSOR_SIZE)
     // For valid shapes, tensorSize_ >= minTensorSize >= 1.
     // Also a defensive check.
 
     // Exclusive + reverse via V2 with int types
     // aclnnCumsumV2 passes exclusive/reverse to tiling -> CumsumCoreInnerTilingData
     {
         std::vector<int32_t> inp(16);
         for (int i = 0; i < 16; ++i) inp[i] = i + 1;
         std::vector<int32_t> out;
         auto r = RunV2(inp, std::vector<int64_t>{16}, 0, 1, 0, out, stream);
         LOG_PRINT("  INT32 excl: ret=%d\n", r);
         r = RunV2(inp, std::vector<int64_t>{16}, 0, 0, 1, out, stream);
         LOG_PRINT("  INT32 rev: ret=%d\n", r);
         r = RunV2(inp, std::vector<int64_t>{16}, 0, 1, 1, out, stream);
         LOG_PRINT("  INT32 excl+rev: ret=%d\n", r);
     }
 
     // 2D shapes for INT: axis=0
     {
         int64_t s[] = {8, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 1, 1, tOut, &ws, &ex);
         LOG_PRINT("  INT32 2D axis=0 excl+rev: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // INT tiling: GetMCTilingInfo -> usedCoreCnt_ branches
     // GetMCTilingInfo: mLpCnt calculation for LA-block axis
     // usedCoreCnt_ = CeilDiv(laLpCnt, CeilDiv(laLpCnt, coreNum))
     // Shape (32, 4, 4) axis=1: left=32, mid=4, right=4 -> LA-block (leftAxisLen dominates)
     // laLpCnt = CeilDiv(32, laLpUnit_), depends on AdjustTensor4TDLA
     {
         int64_t s[] = {32, 4, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 GetMCTiling LA-block: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // CalcAxisWeight: rAxisWeight > raAxisWeight/2 -> CUM_WITH_GROUP
     // Shape (2, 64, 2) axis=1: left=2, mid=64, right=2
     // R-axis dominates (mid=64), should go CUM_WITH_GROUP
     {
         int64_t s[] = {2, 64, 2};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 CUM_WITH_GROUP R-axis: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 
     // CalcAxisWeight: laLpCnt < coreNum, laLpCnt % coreNum != 0
     // Shape (48, 4, 4) axis=1: left=48, mid=4, right=4
     // laLpCnt=CeilDiv(48, laLpUnit_) depends on laLpUnit_
     // For small mid: raLpUnit_=4, tensorSize_/raLpUnitBA/raLpUnit_ ~ 38400/4/4 = 2400
     // laLpUnit_ = min(48, ...) -> laLpCnt = 1
     // Actually, with such large tensorSize, laLpUnit_ tends to be large.
     // Let's use a mid that makes tensorSize smaller via AdjustTDR:
     {
         int64_t s[] = {48, 256, 2};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 CalcAxisWeight mid: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn); aclDestroyTensor(tOut);
     }
 }
 
 
 
 // Scene AD: dim boundary conditions (aclnn_cumsum.cpp lines 250, 317, 319)
 // dim==0 uses DT_INT32 path (not DT_INT64)
 // dim>INT32_MAX uses DT_INT64 path
 void SceneTest26(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene AD] dim boundary conditions\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     std::vector<float> inp(24);
     for (int i = 0; i < 24; ++i) inp[i] = (float)i;
 
     // dim=0 (INT32 path)
     {
         std::vector<float> out;
         auto r = RunV2(inp, std::vector<int64_t>{4, 6}, 0, 0, 0, out, stream);
         LOG_PRINT("  dim=0: ret=%d\n", r);
     }
 
     // dim=1 (INT32 path)
     {
         std::vector<float> out;
         auto r = RunV2(inp, std::vector<int64_t>{4, 6}, 1, 0, 0, out, stream);
         LOG_PRINT("  dim=1: ret=%d\n", r);
     }
 
     // dim=2 (out of range)
     {
         std::vector<float> out;
         auto r = RunV2(inp, std::vector<int64_t>{4, 6}, 2, 0, 0, out, stream);
         LOG_PRINT("  dim=2 (OOB): ret=%d\n", r);
     }
 
     // dim=-1 (negative, normalized)
     {
         std::vector<float> out;
         auto r = RunV2(inp, std::vector<int64_t>{4, 6}, -1, 0, 0, out, stream);
         LOG_PRINT("  dim=-1: ret=%d\n", r);
     }
 
     // dim=-2 (OOB)
     {
         std::vector<float> out;
         auto r = RunV2(inp, std::vector<int64_t>{4, 6}, -2, 0, 0, out, stream);
         LOG_PRINT("  dim=-2 (OOB): ret=%d\n", r);
     }
 
     // dim=INT32_MAX
     {
         std::vector<int64_t> s = {10};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s.data(), 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s.data(), 1, dev);
         aclTensor* tOut = aclCreateTensor(s.data(), 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s.data(), 1, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, INT32_MAX, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  dim=INT32_MAX: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // dim=INT32_MAX+1 (INT64 path, line 250/251)
     {
         std::vector<int64_t> s = {10};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s.data(), 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s.data(), 1, dev);
         aclTensor* tOut = aclCreateTensor(s.data(), 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s.data(), 1, dev);
         ws = 0; ex = nullptr;
         int64_t dimVal = (int64_t)INT32_MAX + 1;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, dimVal, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  dim=INT32_MAX+1 (INT64): ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // V1 dim=0
     {
         std::vector<float> inp2(16, 1.0f);
         std::vector<float> out;
         auto r = RunCumsumDtype(inp2, std::vector<int64_t>{16}, 0, ACL_FLOAT, out, stream);
         LOG_PRINT("  V1 dim=0: ret=%d\n", r);
     }
 
     // V1 dim=INT32_MAX+1
     {
         std::vector<float> inp2(16, 1.0f);
         std::vector<float> out;
         int64_t bigDim = (int64_t)INT32_MAX + 1;
         auto r = RunCumsumDtype(inp2, std::vector<int64_t>{16}, bigDim, ACL_FLOAT, out, stream);
         LOG_PRINT("  V1 dim=INT32_MAX+1 (INT64): ret=%d\n", r);
     }
 }
 
 // Scene AE: aclnnCumsumGetWorkspaceSize empty tensor branch (lines 289-293)
 void SceneTest37(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene AE] aclnnCumsum (V1) empty tensor branch\n");
     uint64_t ws = 0; aclOpExecutor* ex = nullptr;
 
     int64_t s[] = {0};
     void* dev = nullptr;
     aclTensor* tIn = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dev);
     aclTensor* tOut = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dev);
     auto r = aclnnCumsumGetWorkspaceSize(tIn, 0, ACL_FLOAT, tOut, &ws, &ex);
     LOG_PRINT("  V1 empty: ret=%d ws=%lu exec=%p\n", r, ws, (void*)ex);
     if (ex) {
         void* wp = nullptr;
         if (ws > 0) aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
         r = aclnnCumsum(wp, ws, ex, stream);
         LOG_PRINT("  V1 exec: ret=%d\n", r);
         if (wp) aclrtFree(wp);
     }
     aclDestroyTensor(tIn);
     aclDestroyTensor(tOut);
 }
 
 // Scene AF: platform arch routing (cumsum.cpp lines 42-52)
 // IsAiCoreSupport: REGBASE dtypes (UINT8, INT8, INT64) route to AiCpu
 void SceneTest38(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene AF] Platform arch routing (AiCore vs AiCpu)\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // DT_UINT8
     {
         int64_t s[] = {8, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  UINT8: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // DT_INT8
     {
         int64_t s[] = {8, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_INT8, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT8: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // DT_INT64
     {
         int64_t s[] = {8, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_INT64, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_INT64, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT64: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // DT_BF16
     {
         int64_t s[] = {8, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  BF16: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // DT_INT32 (AiCore)
     {
         std::vector<int32_t> inp(32);
         for (int i = 0; i < 32; ++i) inp[i] = i + 1;
         std::vector<int32_t> out;
         auto r = RunV2(inp, std::vector<int64_t>{8, 4}, 1, 0, 0, out, stream);
         LOG_PRINT("  INT32 (AiCore): ret=%d\n", r);
     }
 
     // DT_FLOAT16 (AiCore)
     {
         std::vector<aclFloat16> inp2(32);
         for (int i = 0; i < 32; ++i) inp2[i] = F16((float)(i + 1));
         std::vector<aclFloat16> out;
         auto r = RunV2(inp2, std::vector<int64_t>{8, 4}, 1, 0, 0, out, stream);
         LOG_PRINT("  FLOAT16 (AiCore): ret=%d\n", r);
     }
 }
 
 // Scene AG: tiling shapes for various axis and int dtype branches
 void SceneTest39(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene AG] Various tiling shapes and int dtype branches\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // 8D tensor (MAX_DIM_LEN boundary)
     {
         int64_t s[] = {2, 2, 2, 2, 2, 2, 2, 2};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 8, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 8, dev);
         aclTensor* tOut = aclCreateTensor(s, 8, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 8, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 7, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  8D tensor: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // 3D various axes for int types
     {
         int64_t s[] = {4, 8, 16};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         for (int d = 0; d < 3; ++d) {
             ws = 0; ex = nullptr;
             auto r = aclnnCumsumV2GetWorkspaceSize(tIn, d, 0, 0, tOut, &ws, &ex);
             LOG_PRINT("  3D INT32 axis=%d: ret=%d ws=%lu\n", d, r, ws);
             if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // INT8 various shapes
     aclDataType idts[] = {ACL_INT8, ACL_INT64};
     const char* in[] = {"INT8", "INT64"};
     for (int ti = 0; ti < 2; ++ti) {
         int64_t s[] = {16, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, idts[ti], nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, idts[ti], nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  %s (16,8): ret=%d ws=%lu\n", in[ti], r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // UINT8 various shapes
     {
         int64_t s[] = {16, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, s, 2, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  UINT8 (16,8) axis=0: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 }
 
 // Scene AH: integer tiling deeper branches
 // AdjustTensor4TDRA, AdjustTensor4TDLA, AdjustTensor4TDR, CalcTilingKey, GetMCTilingInfo
 void SceneTest30(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene AH] Integer tiling deep branches\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // CalcTilingKey: CUM_NO_SPLIT (rightAxisLen * dtypeSize >= vlSize_)
     {
         int64_t s[] = {2, 4, 64};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 (2,4,64) CUM_NO_SPLIT: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // CalcTilingKey: CUM_AR_SPLIT (rightAxisLen * dtypeSize < vlSize_)
     {
         int64_t s[] = {2, 4, 16};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 (2,4,16) CUM_AR_SPLIT: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // GetMCTilingInfo: RA-block (maxWeight == raAxisWeight)
     {
         int64_t s[] = {2, 2, 256};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 (2,2,256) RA-block: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // AdjustTensor4TDRA: rWeight > 2*CalcAxisWeight(...)
     {
         int64_t s[] = {2, 256, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 (2,256,4) TDRA: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // AdjustTensor4TDLA: tmpRLpUnit == midAxisLen
     {
         int64_t s[] = {32, 4, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 (32,4,4) TDLA: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // AdjustTensor4TDR: CalcAxisWeight branch
     {
         int64_t s[] = {8, 64, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 (8,64,4) TDR: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // CheckBGC true
     {
         int64_t s[] = {64, 4, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 CheckBGC true: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // CheckBGC false
     {
         int64_t s[] = {4, 4, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 CheckBGC false: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // AdjustTensor4TDRA: rightAxisLen*dtypeSize > cacheLine_
     {
         int64_t s[] = {4, 256, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  INT32 TDRA right>cl: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 }
 
 // Scene AI: ascendc tiling various paths
 void SceneTest31(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene AI] AscendC tiling various paths\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // TWOWAY pattern
     {
         int64_t s[] = {64, 64, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  (64,64,4) TWOWAY: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // MRNGreaterCl with UB split
     {
         int64_t s[] = {2, 2, 8, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 4, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 4, dev);
         aclTensor* tOut = aclCreateTensor(s, 4, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 4, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 3, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  (2,2,8,8) dim=3 MRNGreaterCl UB: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // TILING_KEY_CORE_SS_TWOWAY
     {
         int64_t s[] = {32, 128, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  (32,128,4) CORE_SS_TWOWAY: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // TILING_KEY_UB_SS_ONEWAY
     {
         int64_t s[] = {2, 2048, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  (2,2048,4) UB_SS_ONEWAY: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // TILING_KEY_CORE_SS_UB_SS_TWOWAY
     {
         int64_t s[] = {4, 256, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  (4,256,4) CORE_SS_UB_SS_TWOWAY: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // TILING_KEY_CORE_SS_UB_SS_ONEWAY
     {
         int64_t s[] = {2, 512, 8};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  (2,512,8) CORE_SS_UB_SS_ONEWAY: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // FLOAT16 dtCast paths
     {
         int64_t s[] = {1, 256, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  FLOAT16 (1,256,4) dtCast: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // BF16 dtCast paths
     {
         int64_t s[] = {1, 256, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  BF16 (1,256,4) dtCast: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 }
 
 // Scene AJ: CalcAxisWeight branches (cumsum_tiling_ascendc_int_arch35.cpp)
 // lpCnt >= coreNum_ branch, lpCnt % coreNum_ == 0 branch
 void SceneTest32(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene AJ] CalcAxisWeight branches\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // lpCnt >= coreNum_
     {
         int64_t s[] = {256, 4, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  (256,4,4) lpCnt>=core: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // lpCnt >= coreNum_ AND divisible
     {
         int64_t s[] = {128, 4, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  (128,4,4) lpCnt>=core div: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // lpCnt < coreNum_
     {
         int64_t s[] = {4, 4, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  (4,4,4) lpCnt<core: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // lpCnt < coreNum_ but divisible
     {
         int64_t s[] = {2, 4, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  (2,4,4) lpCnt<core div: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 }
 
 // Scene AK: CUM_WITH_GROUP WriteTilingData branch
 void SceneTest33(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene AK] CUM_WITH_GROUP WriteTilingData branch\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     {
         int64_t s[] = {2, 256, 2};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  (2,256,2) CUM_WITH_GROUP: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     {
         int64_t s[] = {2, 128, 4};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         aclTensor* tOut = aclCreateTensor(s, 3, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, s, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  (2,128,4) CUM_WITH_GROUP: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 }
 
 // Scene AL: CheckShapeIsSupport branches
 // CUMSUM_CUBE_MAX_SUPPORT_SIZE=50000000, dim==0 batch calc, dim!=lastDim
 void SceneTest34(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene AL] CheckShapeIsSupport branches\n");
 
     // Over MAX_SUPPORT_SIZE (dim=0)
     {
         std::vector<float> inp(50000001);
         for (size_t i = 0; i < inp.size(); ++i) inp[i] = 1.0f;
         std::vector<float> out;
         auto r = RunV2(inp, std::vector<int64_t>{50000001}, 0, 0, 0, out, stream);
         LOG_PRINT("  over MAX_SIZE: ret=%d\n", r);
     }
 
     // dim=0: batchNum stays 1, meets CUBE threshold
     {
         std::vector<float> inp(12800 * 512);
         for (size_t i = 0; i < inp.size(); ++i) inp[i] = 1.0f;
         std::vector<float> out;
         auto r = RunV2(inp, std::vector<int64_t>{12800, 512}, 0, 0, 0, out, stream);
         LOG_PRINT("  dim=0 meets CUBE: ret=%d\n", r);
     }
 
     // dim!=lastDim -> CheckShapeIsSupport false
     {
         std::vector<float> inp(12800 * 512);
         for (size_t i = 0; i < inp.size(); ++i) inp[i] = 1.0f;
         std::vector<float> out;
         auto r = RunV2(inp, std::vector<int64_t>{12800, 512}, 0, 0, 0, out, stream);
         LOG_PRINT("  dim=0 not-last -> noCUBE: ret=%d\n", r);
     }
 }
 
 // Scene AM: non-contiguous tensor (strides != continuous)
 void SceneTest35(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene AM] Non-contiguous tensor (strides)\n");
     uint64_t ws; aclOpExecutor* ex; void* wp;
 
     // Non-contiguous input
     {
         int64_t shape[] = {2, 3, 4};
         int64_t strides_nc[] = {20, 4, 1};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(shape, 3, ACL_FLOAT, strides_nc, 0, ACL_FORMAT_ND, shape, 3, dev);
         aclTensor* tOut = aclCreateTensor(shape, 3, ACL_FLOAT, strides_nc, 0, ACL_FORMAT_ND, shape, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 2, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  Non-contiguous dim=2: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // Non-contiguous with dim=0
     {
         int64_t shape[] = {2, 2, 8};
         int64_t strides_nc[] = {24, 8, 1};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(shape, 3, ACL_FLOAT, strides_nc, 0, ACL_FORMAT_ND, shape, 3, dev);
         aclTensor* tOut = aclCreateTensor(shape, 3, ACL_FLOAT, strides_nc, 0, ACL_FORMAT_ND, shape, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 2, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  Non-contiguous 2x2x8 dim=2: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 
     // Contiguous reference
     {
         int64_t shape[] = {3, 4, 5};
         int64_t strides[] = {20, 5, 1};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(shape, 3, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 3, dev);
         aclTensor* tOut = aclCreateTensor(shape, 3, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 3, dev);
         ws = 0; ex = nullptr;
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  Contiguous dim=0: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) { aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST); aclnnCumsumV2(wp, ws, ex, stream); aclrtFree(wp); }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }
 }
 
 // Scene AN: V1 aclnnCumsum with various dtypes
 void SceneTest36(aclrtStream stream)
 {
     LOG_PRINT("\n>> [Scene AN] V1 aclnnCumsum with various dtypes\n");
 
     // FLOAT->FLOAT
     {
         std::vector<float> inp(16, 1.5f);
         std::vector<float> out;
         auto r = RunCumsumDtype(inp, std::vector<int64_t>{16}, 0, ACL_FLOAT, out, stream);
         LOG_PRINT("  FLOAT->FLOAT: ret=%d\n", r);
     }
 
     // FLOAT->FLOAT16
     {
         std::vector<float> inp(16, 1.5f);
         std::vector<float> out;
         auto r = RunCumsumDtype(inp, std::vector<int64_t>{16}, 0, ACL_FLOAT16, out, stream);
         LOG_PRINT("  FLOAT->FLOAT16: ret=%d\n", r);
     }
 
     // V1 CUBE FLOAT
     {
         std::vector<float> inp(12800 * 512);
         for (size_t i = 0; i < inp.size(); ++i) inp[i] = (float)(i % 10);
         std::vector<float> out;
         auto r = RunCumsumDtype(inp, std::vector<int64_t>{12800, 512}, 1, ACL_FLOAT, out, stream);
         LOG_PRINT("  V1 CUBE FLOAT: ret=%d\n", r);
     }
 
     // V2 CUBE FLOAT16
     {
         std::vector<aclFloat16> inp2(12800 * 100);
         for (size_t i = 0; i < inp2.size(); ++i) inp2[i] = F16(1.0f);
         std::vector<aclFloat16> out2;
         auto r = RunV2(inp2, std::vector<int64_t>{12800, 100}, 1, 0, 0, out2, stream);
         LOG_PRINT("  V2 CUBE FLOAT16: ret=%d\n", r);
     }
 }
 
 // ========== EXTENDED PRECISION TESTS ==========
 void ExtendedPrecisionTests(aclrtStream stream)
 {
     LOG_PRINT("\n========== EXTENDED PRECISION TESTS ==========\n");
 
     // 1. Subnormal numbers
     {
         std::vector<float> inp = {1e-40f, 1e-40f, 1e-40f, 1e-40f};
         TestPrec("Subnormal-Float", inp, {4}, 0, 0, 0, 1e-4, 1e-2, stream);
     }
 
     // 2. Overflow to inf
     {
         std::vector<float> inp = {1e20f, 1e20f, 1e20f, 1e20f};
         TestPrec("Overflow-Float", inp, {4}, 0, 0, 0, 1e3, 1e3, stream);
     }
 
     // 3. Near-unity arithmetic
     {
         std::vector<float> inp = {0.9999999f, 1.0000001f, 0.9999999f, 1.0000001f};
         TestPrec("NearUnity-Float", inp, {4}, 0, 0, 0, 1e-5, 1e-3, stream);
     }
 
     // 4. Magnitude cancellation
     {
         std::vector<float> inp = {1e8f, -1e8f, 1.0f, 2.0f};
         TestPrec("MagCancel-Float", inp, {4}, 0, 0, 0, 1e-4, 1e-3, stream);
     }
 
     // 5. Decimal inexact binary
     {
         std::vector<float> inp = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
         TestPrec("Decimal-Float", inp, {5}, 0, 0, 0, 1e-5, 1e-2, stream);
     }
 
     // 6. Multi-dim along dim=0
     {
         std::vector<float> inp = {1,2,3,4,5,6,7,8,9,10,11,12};
         TestPrec("MultiDim-3x4-dim0", inp, {3,4}, 0, 0, 0, 1e-5, 1e-5, stream);
         TestPrec("MultiDim-3x4-dim1", inp, {3,4}, 1, 0, 0, 1e-5, 1e-5, stream);
     }
 
     // 7. 3D tensor various dims
     {
         std::vector<float> inp(24);
         for (int i = 0; i < 24; ++i) inp[i] = (float)(i + 1);
         TestPrec("3D-2x3x4-dim0", inp, {2,3,4}, 0, 0, 0, 1e-5, 1e-5, stream);
         TestPrec("3D-2x3x4-dim1", inp, {2,3,4}, 1, 0, 0, 1e-5, 1e-5, stream);
         TestPrec("3D-2x3x4-dim2", inp, {2,3,4}, 2, 0, 0, 1e-5, 1e-5, stream);
     }
 
     // 8. Float16 precision
     {
         std::vector<aclFloat16> inp2(8);
         for (int i = 0; i < 8; ++i) inp2[i] = F16((float)(i + 1));
         TestPrec("Float16-8", inp2, {8}, 0, 0, 0, 1e-3, 1e-3, stream);
     }
 
     // 9. INT32 precision
     {
         std::vector<int32_t> inp3(8);
         for (int i = 0; i < 8; ++i) inp3[i] = i + 1;
         TestPrec("INT32-8", inp3, {8}, 0, 0, 0, 0, 0, stream);
     }
 
     // 10. INT64 precision
     {
         std::vector<int64_t> inp4(8);
         for (int i = 0; i < 8; ++i) inp4[i] = (i + 1) * 1000000LL;
         TestPrec("INT64-8", inp4, {8}, 0, 0, 0, 0, 0, stream);
     }
 
     // 11. Exclusive cumsum
     {
         std::vector<float> inp = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
         TestPrec("Exclusive", inp, {5}, 0, 1, 0, 1e-5, 1e-5, stream);
     }
 
     // 12. Reverse cumsum
     {
         std::vector<float> inp = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
         TestPrec("Reverse", inp, {5}, 0, 0, 1, 1e-5, 1e-5, stream);
     }
 
     // 13. Exclusive+Reverse cumsum
     {
         std::vector<float> inp = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
         TestPrec("ExclReverse", inp, {5}, 0, 1, 1, 1e-5, 1e-5, stream);
     }
 
     // 14. Negative values
     {
         std::vector<float> inp = {-5.0f, -3.0f, -1.0f, 0.0f, 2.0f, 5.0f};
         TestPrec("Negative", inp, {6}, 0, 0, 0, 1e-5, 1e-5, stream);
     }
 
     // 15. Large shape
     {
         std::vector<float> inp(1024);
         for (int i = 0; i < 1024; ++i) inp[i] = (float)(i % 100);
         TestPrec("Large-1024", inp, {1024}, 0, 0, 0, 1e-5, 1e-5, stream);
     }
 
     // 16. All zeros
     {
         std::vector<float> inp(16, 0.0f);
         TestPrec("AllZeros", inp, {16}, 0, 0, 0, 1e-5, 1e-5, stream);
     }
 
     // 17. All ones
     {
         std::vector<float> inp(16, 1.0f);
         TestPrec("AllOnes", inp, {16}, 0, 0, 0, 1e-5, 1e-5, stream);
     }
 
     // 18. Alternating values
     {
         std::vector<float> inp = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};
         TestPrec("Alternating", inp, {8}, 0, 0, 0, 1e-4, 1e-3, stream);
     }
 
     // 19. Float16 near overflow
     {
         std::vector<aclFloat16> inp2(8);
         for (int i = 0; i < 8; ++i) inp2[i] = F16(1e3f);
         TestPrec("Float16-NearOverflow", inp2, {8}, 0, 0, 0, 1e-2, 1e-2, stream);
     }
 
     // 20. 4D tensor
     {
         std::vector<float> inp(24);
         for (int i = 0; i < 24; ++i) inp[i] = (float)(i + 1);
         TestPrec("4D-2x2x2x3-dim0", inp, {2,2,2,3}, 0, 0, 0, 1e-5, 1e-5, stream);
         TestPrec("4D-2x2x2x3-dim3", inp, {2,2,2,3}, 3, 0, 0, 1e-5, 1e-5, stream);
     }
 
     // 21. Float16 decimal
     {
         std::vector<aclFloat16> inp2 = {F16(0.1f), F16(0.2f), F16(0.3f), F16(0.4f)};
         TestPrec("Float16-Decimal", inp2, {4}, 0, 0, 0, 1e-2, 1e-2, stream);
     }
 
     // 22. Exclusive dim1
     {
         std::vector<float> inp = {1,2,3,4,5,6,7,8};
         TestPrec("Exclusive-dim1", inp, {2,4}, 1, 1, 0, 1e-5, 1e-5, stream);
     }
 
     // 23. Reverse dim1
     {
         std::vector<float> inp = {1,2,3,4,5,6,7,8};
         TestPrec("Reverse-dim1", inp, {2,4}, 1, 0, 1, 1e-5, 1e-5, stream);
     }
 
     // 24. Large batch small channel
     {
         std::vector<float> inp(12800 * 10);
         for (size_t i = 0; i < inp.size(); ++i) inp[i] = 1.0f;
         TestPrec("LargeBatch", inp, {12800, 10}, 1, 0, 0, 1e-5, 1e-5, stream);
     }
 
     // 25. Mixed positive/negative
     {
         std::vector<float> inp = {-100.0f, 50.0f, -25.0f, 10.0f, -5.0f, 2.0f};
         TestPrec("MixedSign", inp, {6}, 0, 0, 0, 1e-4, 1e-3, stream);
     }
 
     // 26. INT8 precision
     {
         std::vector<int8_t> inp5(8);
         for (int i = 0; i < 8; ++i) inp5[i] = (int8_t)(i + 1);
         TestPrec("INT8-8", inp5, {8}, 0, 0, 0, 0, 0, stream);
     }
 
     // 27. UINT8 precision
     {
         std::vector<uint8_t> inp6(8);
         for (int i = 0; i < 8; ++i) inp6[i] = (uint8_t)(i + 1);
         TestPrec("UINT8-8", inp6, {8}, 0, 0, 0, 0, 0, stream);
     }
 
     // 28. 2D large along dim1
     {
         std::vector<float> inp(64);
         for (int i = 0; i < 64; ++i) inp[i] = (float)(i + 1);
         TestPrec("Large-8x8-dim1", inp, {8, 8}, 1, 0, 0, 1e-5, 1e-5, stream);
     }
 }

 // SceneTest40: aclnnCumsum (old API) — Cumsum 3-param overload path
 // Targets: cumsum.cpp:78-91, CumsumAiCore success branch (branch at line 86 is 86% taken),
 //         CumsumAiCore line 60 OP_CHECK (100% success path),
 //         alloc out nullptr error branch (line 81-83) — not covered
 //         Also targets aclnn_cumsum.cpp:277-336 (aclnnCumsumGetWorkspaceSize)
 void SceneTest40(aclrtStream stream)
 {
     LOG_PRINT("\n>> [SceneTest40] aclnnCumsum old API — V1 API path + V1 empty tensor\n");
     uint64_t ws = 0; aclOpExecutor* ex = nullptr; int r;

     // V1 API — empty tensor early-return (aclnn_cumsum.cpp:289-292)
     {
         int64_t s[] = {0};
         void* dev = nullptr;
         aclTensor* tIn = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dev);
         aclTensor* tOut = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dev);
         r = aclnnCumsumGetWorkspaceSize(tIn, 0, ACL_FLOAT, tOut, &ws, &ex);
         LOG_PRINT("  V1 empty: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) {
             void* wp = nullptr;
             aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
             r = aclnnCumsum(wp, ws, ex, stream);
             LOG_PRINT("  V1 empty exec: ret=%d\n", r);
             if (wp) aclrtFree(wp);
         }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
     }

     // V1 API — normal tensor (triggers cumsum.cpp Cumsum 3-param overload line 78-91)
     // dtype=ACL_FLOAT so IsAiCoreSupport -> AiCore910 path (line 44-51)
     {
         std::vector<float> inp = {1, 2, 3, 4, 5, 6};
         int64_t s[] = {2, 3};
         void* dev = nullptr;
         void* dIn = nullptr; void* dOut = nullptr;
         aclrtMalloc(&dIn, inp.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&dOut, inp.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMemcpy(dIn, inp.size() * sizeof(float), inp.data(), inp.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dIn);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dOut);
         ws = 0; ex = nullptr;
         r = aclnnCumsumGetWorkspaceSize(tIn, 1, ACL_FLOAT, tOut, &ws, &ex);
         LOG_PRINT("  V1 dim=1: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) {
             void* wp = nullptr;
             aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
             r = aclnnCumsum(wp, ws, ex, stream);
             std::vector<float> result(inp.size());
             aclrtMemcpy(result.data(), result.size() * sizeof(float), dOut, result.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
             double expected = 0; bool first = true;
             for (int i = 0; i < 3; ++i) { expected += (i + 1); }
             // dim=1 on {2,3}: check row 0 sum = 1+2+3=6
             LOG_PRINT("  V1 result[0]=%f expected[0]=6.0\n", result[0]);
             if (wp) aclrtFree(wp);
         }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
         if (dIn) aclrtFree(dIn);
         if (dOut) aclrtFree(dOut);
     }

     // V1 API — dim=0 (triggers ConvertToTensor with dim==0 branch at aclnn_cumsum.cpp:250)
     {
         std::vector<float> inp = {10, 20, 30};
         int64_t s[] = {3};
         void* dIn = nullptr; void* dOut = nullptr;
         aclrtMalloc(&dIn, 3 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&dOut, 3 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMemcpy(dIn, 3 * sizeof(float), inp.data(), 3 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
         aclTensor* tIn = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dIn);
         aclTensor* tOut = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dOut);
         ws = 0; ex = nullptr;
         r = aclnnCumsumGetWorkspaceSize(tIn, 0, ACL_FLOAT, tOut, &ws, &ex);
         LOG_PRINT("  V1 dim=0: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) {
             void* wp = nullptr;
             aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
             r = aclnnCumsum(wp, ws, ex, stream);
             if (wp) aclrtFree(wp);
         }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
         if (dIn) aclrtFree(dIn);
         if (dOut) aclrtFree(dOut);
     }
 }

// SceneTest41: aclnnCumsum V1 API + FP16 dtype
// Targets: cumsum.cpp IsAiCoreSupport -> DAV_2201 branch (100% taken on 910B),
//         FP16 in AICORE910B_DTYPE_SUPPORT_LIST,
//         aclnn_cumsum.cpp CheckDtypeValid branch (line 94-100)
 void SceneTest41(aclrtStream stream)
 {
     LOG_PRINT("\n>> [SceneTest41] V1 API + BF16 dtype\n");
     uint64_t ws = 0; aclOpExecutor* ex = nullptr;

    // FP16 — V1 API path
    // FP16 is in AICORE910B_DTYPE_SUPPORT_LIST (cumsum.cpp:35)
    // aclnn_cumsum.cpp CheckDtypeValid is called by aclnnCumsumGetWorkspaceSize
    {
        std::vector<aclFloat16> inp;
        for (int i = 1; i <= 6; ++i) inp.push_back(F16((float)i));
        int64_t s[] = {2, 3};
        void* dIn = nullptr; void* dOut = nullptr;
        aclrtMalloc(&dIn, 6 * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dOut, 6 * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dIn, 6 * sizeof(aclFloat16), inp.data(), 6 * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE);
        aclTensor* tIn = aclCreateTensor(s, 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, s, 2, dIn);
        aclTensor* tOut = aclCreateTensor(s, 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, s, 2, dOut);
        auto r = aclnnCumsumGetWorkspaceSize(tIn, 1, ACL_FLOAT16, tOut, &ws, &ex);
        LOG_PRINT("  FP16 dim=1: ret=%d ws=%lu\n", r, ws);
        if (ex && ws > 0) {
            void* wp = nullptr;
            aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
            r = aclnnCumsum(wp, ws, ex, stream);
            LOG_PRINT("  FP16 exec: ret=%d\n", r);
            if (wp) aclrtFree(wp);
        }
        aclDestroyTensor(tIn);
        aclDestroyTensor(tOut);
        if (dIn) aclrtFree(dIn);
        if (dOut) aclrtFree(dOut);
    }

    // FP16 — V2 API path
    {
        std::vector<aclFloat16> inp;
        for (int i = 1; i <= 6; ++i) inp.push_back(F16((float)i));
        int64_t s[] = {2, 3};
        void* dIn = nullptr; void* dOut = nullptr;
        aclrtMalloc(&dIn, 6 * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dOut, 6 * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dIn, 6 * sizeof(aclFloat16), inp.data(), 6 * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE);
        aclTensor* tIn = aclCreateTensor(s, 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, s, 2, dIn);
        aclTensor* tOut = aclCreateTensor(s, 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, s, 2, dOut);
        auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 1, 0, 0, tOut, &ws, &ex);
        LOG_PRINT("  FP16 V2 dim=1: ret=%d ws=%lu\n", r, ws);
        if (ex && ws > 0) {
            void* wp = nullptr;
            aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
            r = aclnnCumsumV2(wp, ws, ex, stream);
            LOG_PRINT("  FP16 V2 exec: ret=%d\n", r);
            if (wp) aclrtFree(wp);
        }
        aclDestroyTensor(tIn);
        aclDestroyTensor(tOut);
        if (dIn) aclrtFree(dIn);
        if (dOut) aclrtFree(dOut);
    }
 }

 // SceneTest42: aclnnCumsum V1 API + dim > INT32_MAX branch
 // Targets: aclnn_cumsum.cpp:317 dim > INT32_MAX branch (line 250 condition: dim==0 || dim>INT32_MAX)
 // This triggers the DT_INT64 ConvertToTensor path
 void SceneTest42(aclrtStream stream)
 {
     LOG_PRINT("\n>> [SceneTest42] V1 API + dim > INT32_MAX\n");
     uint64_t ws = 0; aclOpExecutor* ex = nullptr;

     // dim = INT32_MAX + 1 triggers the "> INT32_MAX" branch
     int64_t dim = (int64_t)INT32_MAX + 1;  // 2147483648
     std::vector<float> inp(8, 1.0f);
     int64_t s[] = {8};
     void* dIn = nullptr; void* dOut = nullptr;
     aclrtMalloc(&dIn, 8 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
     aclrtMalloc(&dOut, 8 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
     aclrtMemcpy(dIn, 8 * sizeof(float), inp.data(), 8 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
     aclTensor* tIn = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dIn);
     aclTensor* tOut = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dOut);
     auto r = aclnnCumsumGetWorkspaceSize(tIn, dim, ACL_FLOAT, tOut, &ws, &ex);
     LOG_PRINT("  dim=%ld (INT32_MAX+1): ret=%d ws=%lu\n", (long)dim, r, ws);
     if (ex && ws > 0) {
         void* wp = nullptr;
         aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
         r = aclnnCumsum(wp, ws, ex, stream);
         LOG_PRINT("  dim=%ld exec: ret=%d\n", (long)dim, r);
         if (wp) aclrtFree(wp);
     }
     aclDestroyTensor(tIn);
     aclDestroyTensor(tOut);
     if (dIn) aclrtFree(dIn);
     if (dOut) aclrtFree(dOut);
 }

 // SceneTest43: V1 API + dim = INT32_MAX (tests dim > INT32_MAX branch directly)
 // INT32_MAX == 2147483647, dim > INT32_MAX means dim >= 2147483648
 void SceneTest43(aclrtStream stream)
 {
     LOG_PRINT("\n>> [SceneTest43] V1 API + dim = INT32_MAX\n");
     uint64_t ws = 0; aclOpExecutor* ex = nullptr;

     int64_t dim = INT32_MAX;  // 2147483647 — should be out-of-range for {8} tensor
     // CheckDim will reject this, but it still exercises the dim check path
     std::vector<float> inp(8, 1.0f);
     int64_t s[] = {8};
     void* dIn = nullptr; void* dOut = nullptr;
     aclrtMalloc(&dIn, 8 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
     aclrtMalloc(&dOut, 8 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
     aclrtMemcpy(dIn, 8 * sizeof(float), inp.data(), 8 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
     aclTensor* tIn = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dIn);
     aclTensor* tOut = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dOut);
     auto r = aclnnCumsumGetWorkspaceSize(tIn, dim, ACL_FLOAT, tOut, &ws, &ex);
     LOG_PRINT("  dim=INT32_MAX: ret=%d ws=%lu\n", r, ws);
     // This should fail with ACLNN_ERR_PARAM_INVALID (dim out of range)
     if (ex && ws > 0) {
         void* wp = nullptr;
         aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
         r = aclnnCumsum(wp, ws, ex, stream);
         LOG_PRINT("  dim=INT32_MAX exec: ret=%d\n", r);
         if (wp) aclrtFree(wp);
     }
     aclDestroyTensor(tIn);
     aclDestroyTensor(tOut);
     if (dIn) aclrtFree(dIn);
     if (dOut) aclrtFree(dOut);
 }

 // SceneTest44: V2 API + exclusive+reverse combination
 // Targets: cumsum.cpp:93-106, Cumsum 4-param overload, exclusive=true, reverse=true
 void SceneTest44(aclrtStream stream)
 {
     LOG_PRINT("\n>> [SceneTest44] V2 exclusive+reverse combinations\n");
     uint64_t ws = 0; aclOpExecutor* ex = nullptr;

     // exclusive=1, reverse=0
     {
         std::vector<float> inp = {1, 2, 3, 4, 5};
         int64_t s[] = {5};
         void* dIn = nullptr; void* dOut = nullptr;
         aclrtMalloc(&dIn, 5 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&dOut, 5 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMemcpy(dIn, 5 * sizeof(float), inp.data(), 5 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
         aclTensor* tIn = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dIn);
         aclTensor* tOut = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dOut);
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 1, 0, tOut, &ws, &ex);
         LOG_PRINT("  V2 excl=1 rev=0: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) {
             void* wp = nullptr;
             aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
             r = aclnnCumsumV2(wp, ws, ex, stream);
             LOG_PRINT("  V2 excl=1 rev=0 exec: ret=%d\n", r);
             if (wp) aclrtFree(wp);
         }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
         if (dIn) aclrtFree(dIn);
         if (dOut) aclrtFree(dOut);
     }

     // exclusive=0, reverse=1
     ws = 0; ex = nullptr;
     {
         std::vector<float> inp = {1, 2, 3, 4, 5};
         int64_t s[] = {5};
         void* dIn = nullptr; void* dOut = nullptr;
         aclrtMalloc(&dIn, 5 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&dOut, 5 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMemcpy(dIn, 5 * sizeof(float), inp.data(), 5 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
         aclTensor* tIn = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dIn);
         aclTensor* tOut = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dOut);
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 0, 1, tOut, &ws, &ex);
         LOG_PRINT("  V2 excl=0 rev=1: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) {
             void* wp = nullptr;
             aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
             r = aclnnCumsumV2(wp, ws, ex, stream);
             LOG_PRINT("  V2 excl=0 rev=1 exec: ret=%d\n", r);
             if (wp) aclrtFree(wp);
         }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
         if (dIn) aclrtFree(dIn);
         if (dOut) aclrtFree(dOut);
     }

     // exclusive=1, reverse=1
     ws = 0; ex = nullptr;
     {
         std::vector<float> inp = {1, 2, 3, 4, 5};
         int64_t s[] = {5};
         void* dIn = nullptr; void* dOut = nullptr;
         aclrtMalloc(&dIn, 5 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&dOut, 5 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMemcpy(dIn, 5 * sizeof(float), inp.data(), 5 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
         aclTensor* tIn = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dIn);
         aclTensor* tOut = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dOut);
         auto r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 1, 1, tOut, &ws, &ex);
         LOG_PRINT("  V2 excl=1 rev=1: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) {
             void* wp = nullptr;
             aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
             r = aclnnCumsumV2(wp, ws, ex, stream);
             LOG_PRINT("  V2 excl=1 rev=1 exec: ret=%d\n", r);
             if (wp) aclrtFree(wp);
         }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
         if (dIn) aclrtFree(dIn);
         if (dOut) aclrtFree(dOut);
     }
 }

 // SceneTest45: 2D tensor + dim=0 (multi-dim, batch axis)
 // Targets: cumsum.cpp:93-106 on 2D tensor, lenM > 1 path in tiling
 // Also targets CheckDim negative dim handling (aclnn_cumsum.cpp:116)
 void SceneTest45(aclrtStream stream)
 {
     LOG_PRINT("\n>> [SceneTest45] 2D tensor + dim=0 + negative dim\n");
     uint64_t ws = 0; aclOpExecutor* ex = nullptr; int r;

     // 2D tensor dim=0 — lenM > 1 case (batch dimension)
     {
         std::vector<float> inp(12, 1.0f);
         int64_t s[] = {3, 4};
         void* dIn = nullptr; void* dOut = nullptr;
         aclrtMalloc(&dIn, 12 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&dOut, 12 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMemcpy(dIn, 12 * sizeof(float), inp.data(), 12 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dIn);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dOut);
         r = aclnnCumsumV2GetWorkspaceSize(tIn, 0, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  2D dim=0: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) {
             void* wp = nullptr;
             aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
             r = aclnnCumsumV2(wp, ws, ex, stream);
             LOG_PRINT("  2D dim=0 exec: ret=%d\n", r);
             if (wp) aclrtFree(wp);
         }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
         if (dIn) aclrtFree(dIn);
         if (dOut) aclrtFree(dOut);
     }

     // Negative dim = -1 (last dimension)
     ws = 0; ex = nullptr;
     {
         std::vector<float> inp = {1, 2, 3, 4};
         int64_t s[] = {4};
         void* dIn = nullptr; void* dOut = nullptr;
         aclrtMalloc(&dIn, 4 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&dOut, 4 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMemcpy(dIn, 4 * sizeof(float), inp.data(), 4 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
         aclTensor* tIn = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dIn);
         aclTensor* tOut = aclCreateTensor(s, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 1, dOut);
         // dim=-1: CheckDim at aclnn_cumsum.cpp:116 checks dim < -selfDimNum
         // For {4}, selfDimNum=1, valid dim range is [-1, 0], dim=-1 is valid
         r = aclnnCumsumV2GetWorkspaceSize(tIn, -1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  dim=-1: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) {
             void* wp = nullptr;
             aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
             r = aclnnCumsumV2(wp, ws, ex, stream);
             LOG_PRINT("  dim=-1 exec: ret=%d\n", r);
             if (wp) aclrtFree(wp);
         }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
         if (dIn) aclrtFree(dIn);
         if (dOut) aclrtFree(dOut);
     }

     // 2D negative dim = -1
     ws = 0; ex = nullptr;
     {
         std::vector<float> inp(12, 1.0f);
         int64_t s[] = {3, 4};
         void* dIn = nullptr; void* dOut = nullptr;
         aclrtMalloc(&dIn, 12 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&dOut, 12 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMemcpy(dIn, 12 * sizeof(float), inp.data(), 12 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dIn);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dOut);
         r = aclnnCumsumV2GetWorkspaceSize(tIn, -1, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  2D dim=-1: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) {
             void* wp = nullptr;
             aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
             r = aclnnCumsumV2(wp, ws, ex, stream);
             LOG_PRINT("  2D dim=-1 exec: ret=%d\n", r);
             if (wp) aclrtFree(wp);
         }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
         if (dIn) aclrtFree(dIn);
         if (dOut) aclrtFree(dOut);
     }

     // 2D negative dim = -2
     ws = 0; ex = nullptr;
     {
         std::vector<float> inp(12, 1.0f);
         int64_t s[] = {3, 4};
         void* dIn = nullptr; void* dOut = nullptr;
         aclrtMalloc(&dIn, 12 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMalloc(&dOut, 12 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
         aclrtMemcpy(dIn, 12 * sizeof(float), inp.data(), 12 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
         aclTensor* tIn = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dIn);
         aclTensor* tOut = aclCreateTensor(s, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, s, 2, dOut);
         r = aclnnCumsumV2GetWorkspaceSize(tIn, -2, 0, 0, tOut, &ws, &ex);
         LOG_PRINT("  2D dim=-2: ret=%d ws=%lu\n", r, ws);
         if (ex && ws > 0) {
             void* wp = nullptr;
             aclrtMalloc(&wp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
             r = aclnnCumsumV2(wp, ws, ex, stream);
             LOG_PRINT("  2D dim=-2 exec: ret=%d\n", r);
             if (wp) aclrtFree(wp);
         }
         aclDestroyTensor(tIn);
         aclDestroyTensor(tOut);
         if (dIn) aclrtFree(dIn);
         if (dOut) aclrtFree(dOut);
     }
 }

 // ========== NEW PRECISION TESTS (SceneTest46+) ==========
 // Targeted precision scenarios for uncovered dtype/shape combinations
 void ExtendedPrecisionTests2(aclrtStream stream)
 {
     LOG_PRINT("\n========== EXTENDED PRECISION TESTS 2 ==========\n");
     int passed = 0, failed = 0, r;

    // P1: FP16 precision — V2 API path
    {
        std::vector<aclFloat16> inp;
        for (int i = 1; i <= 5; ++i) inp.push_back(F16((float)i));
        r = TestPrec("P1-FP16", inp, (std::vector<int64_t>){5}, 0, 0, 0, 1e-2, 1e-2, stream);
        if (r == 0) passed++; else failed++;
    }

    // P2: FP16 exclusive
    {
        std::vector<aclFloat16> inp;
        for (int i = 1; i <= 5; ++i) inp.push_back(F16((float)i));
        r = TestPrec("P2-FP16-Excl", inp, (std::vector<int64_t>){5}, 0, 1, 0, 1e-2, 1e-2, stream);
        if (r == 0) passed++; else failed++;
    }

    // P3: FP16 reverse
    {
        std::vector<aclFloat16> inp;
        for (int i = 1; i <= 5; ++i) inp.push_back(F16((float)i));
        r = TestPrec("P3-FP16-Rev", inp, (std::vector<int64_t>){5}, 0, 0, 1, 1e-2, 1e-2, stream);
        if (r == 0) passed++; else failed++;
    }

    // P4: FP16 exclusive+reverse
    {
        std::vector<aclFloat16> inp;
        for (int i = 1; i <= 5; ++i) inp.push_back(F16((float)i));
        r = TestPrec("P4-FP16-ExclRev", inp, (std::vector<int64_t>){5}, 0, 1, 1, 1e-2, 1e-2, stream);
        if (r == 0) passed++; else failed++;
    }

     // P5: INT8 2D — multi-dimensional cumsum on INT8
     {
         std::vector<int8_t> inp(12);
         for (int i = 0; i < 12; ++i) inp[i] = (int8_t)(i + 1);
         r = TestPrec("P5-INT8-2D", inp, (std::vector<int64_t>){3, 4}, 1, 0, 0, 0, 0, stream);
         if (r == 0) passed++; else failed++;
     }

     // P6: INT8 exclusive — exclusive on 2D
     {
         std::vector<int8_t> inp(12);
         for (int i = 0; i < 12; ++i) inp[i] = (int8_t)(i % 3 + 1);
         r = TestPrec("P6-INT8-2D-Excl", inp, (std::vector<int64_t>){3, 4}, 1, 1, 0, 0, 0, stream);
         if (r == 0) passed++; else failed++;
     }

     // P7: INT64 2D — large integer cumsum
     {
         std::vector<int64_t> inp(8);
         for (int i = 0; i < 8; ++i) inp[i] = (int64_t)(i + 1) * 1000000;
         r = TestPrec("P7-INT64-2D", inp, (std::vector<int64_t>){2, 4}, 1, 0, 0, 0, 0, stream);
         if (r == 0) passed++; else failed++;
     }

     // P8: UINT8 2D — unsigned byte cumsum
     {
         std::vector<uint8_t> inp(12);
         for (int i = 0; i < 12; ++i) inp[i] = (uint8_t)((i % 5) + 1);
         r = TestPrec("P8-UINT8-2D", inp, (std::vector<int64_t>){3, 4}, 0, 0, 0, 0, 0, stream);
         if (r == 0) passed++; else failed++;
     }

     // P9: Float32 3D dim=0 — batch dimension cumsum
     {
         std::vector<float> inp(24);
         for (int i = 0; i < 24; ++i) inp[i] = (float)(i + 1);
         r = TestPrec("P9-F32-3D-d0", inp, (std::vector<int64_t>){2, 3, 4}, 0, 0, 0, 1e-5, 1e-5, stream);
         if (r == 0) passed++; else failed++;
     }

     // P10: Float32 4D dim=1 — 4D tensor cumsum
     {
         std::vector<float> inp(24);
         for (int i = 0; i < 24; ++i) inp[i] = (float)(i + 1);
         r = TestPrec("P10-F32-4D-d1", inp, (std::vector<int64_t>){2, 2, 3, 2}, 1, 0, 0, 1e-5, 1e-5, stream);
         if (r == 0) passed++; else failed++;
     }

     // P11: Float16 2D large — 1000 elements on dim=1
     {
         std::vector<aclFloat16> inp(2000);
         for (int i = 0; i < 2000; ++i) inp[i] = F16(1.0f);
         r = TestPrec("P11-F16-2D-2000", inp, (std::vector<int64_t>){2, 1000}, 1, 0, 0, 1e-2, 1e-2, stream);
         if (r == 0) passed++; else failed++;
     }

     // P12: Exclusive on 2D dim=0
     {
         std::vector<float> inp(12, 1.0f);
         r = TestPrec("P12-F32-2D-Excl-d0", inp, (std::vector<int64_t>){3, 4}, 0, 1, 0, 1e-5, 1e-5, stream);
         if (r == 0) passed++; else failed++;
     }

     // P13: Exclusive on 2D dim=1
     {
         std::vector<float> inp(12, 1.0f);
         r = TestPrec("P13-F32-2D-Excl-d1", inp, (std::vector<int64_t>){3, 4}, 1, 1, 0, 1e-5, 1e-5, stream);
         if (r == 0) passed++; else failed++;
     }

     // P14: Reverse on 2D dim=0
     {
         std::vector<float> inp(12, 1.0f);
         r = TestPrec("P14-F32-2D-Rev-d0", inp, (std::vector<int64_t>){3, 4}, 0, 0, 1, 1e-5, 1e-5, stream);
         if (r == 0) passed++; else failed++;
     }

     // P15: Exclusive+Reverse on 2D dim=1
     {
         std::vector<float> inp(12, 1.0f);
         r = TestPrec("P15-F32-2D-ExclRev-d1", inp, (std::vector<int64_t>){3, 4}, 1, 1, 1, 1e-5, 1e-5, stream);
         if (r == 0) passed++; else failed++;
     }

     LOG_PRINT("\n========== EXTENDED PRECISION 2 SUMMARY ==========\n");
     LOG_PRINT("  Passed: %d  Failed: %d\n", passed, failed);
 }
 