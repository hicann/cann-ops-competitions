/**
 * Add 算子端到端测试用例 - 龙湖小队
 * 覆盖：aclnnAdd/aclnnAdds/aclnnInplaceAdd/aclnnInplaceAdds/aclnnAddV3/aclnnInplaceAddV3
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

#define CHECK_RET(cond, return_expr) do { if (!(cond)) { return_expr; } } while (0)
#define LOG_PRINT(message, ...) do { printf(message, ##__VA_ARGS__); } while (0)
#define TEST_CASE(name) do { printf("\n[TEST] %s\n", name); } while(0)
#define EXPECT_SUCCESS(ret, api) \
  do { if ((ret) != ACL_SUCCESS) { printf("[FAIL] %s returned %d\n", api, (int)(ret)); g_failCount++; } \
       else { printf("[PASS] %s\n", api); g_passCount++; } } while(0)
#define EXPECT_FAIL(ret, api) \
  do { if ((ret) == ACL_SUCCESS) { printf("[FAIL] %s should have failed\n", api); g_failCount++; } \
       else { printf("[PASS] %s (expected fail) ret=%d\n", api, (int)(ret)); g_passCount++; } } while(0)
#define VERIFY(ok, name) \
  do { if (ok) { printf("[PASS] verify: %s\n", name); g_passCount++; } \
       else    { printf("[FAIL] verify: %s\n", name); g_failCount++; } } while(0)

static int g_passCount = 0, g_failCount = 0;

static int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t s = 1; for (auto d : shape) s *= d; return s;
}

static int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
static int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                            void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
  int64_t numElem = GetShapeSize(shape);
  size_t size = numElem * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size > 0 ? size : 1, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  if (size > 0) {
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  }
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--)
    strides[i] = shape[i + 1] * strides[i + 1];
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

static aclnnStatus RunAdd(aclTensor* self, aclTensor* other, aclScalar* alpha,
                           aclTensor* out, aclrtStream stream) {
  uint64_t ws = 0; aclOpExecutor* ex = nullptr;
  auto ret = aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &ex);
  if (ret != ACL_SUCCESS) return ret;
  void* wsp = nullptr;
  if (ws > 0) { ret = aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST); if (ret != ACL_SUCCESS) return ret; }
  ret = aclnnAdd(wsp, ws, ex, stream);
  aclrtSynchronizeStream(stream);
  if (wsp) aclrtFree(wsp);
  return ret;
}

static bool VerifyFloat(void* devOut, const std::vector<float>& expected, float atol, float rtol) {
  std::vector<float> actual(expected.size(), 0.f);
  aclrtMemcpy(actual.data(), actual.size() * sizeof(float), devOut,
              actual.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  for (size_t i = 0; i < expected.size(); i++) {
    float e = expected[i], a = actual[i];
    if (std::isnan(e) && std::isnan(a)) continue;
    if (std::isinf(e) && std::isinf(a) && std::signbit(e) == std::signbit(a)) continue;
    if (std::abs(a - e) > atol + rtol * std::abs(e)) {
      printf("    idx=%zu actual=%f expected=%f\n", i, a, e); return false;
    }
  }
  return true;
}

static bool VerifyInt32(void* devOut, const std::vector<int32_t>& expected) {
  std::vector<int32_t> actual(expected.size(), 0);
  aclrtMemcpy(actual.data(), actual.size() * sizeof(int32_t), devOut,
              actual.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  for (size_t i = 0; i < expected.size(); i++) {
    if (actual[i] != expected[i]) {
      printf("    idx=%zu actual=%d expected=%d\n", i, actual[i], expected[i]); return false;
    }
  }
  return true;
}

static void TestTilingDtypes(aclrtStream stream) {
  static float f1=1.f; aclScalar* alpha1f = aclCreateScalar(&f1, ACL_FLOAT);
  static int32_t i1=1; aclScalar* alpha1i = aclCreateScalar(&i1, ACL_INT32);
  static int8_t  i8=1; aclScalar* alpha1i8 = aclCreateScalar(&i8, ACL_INT8);
  static uint8_t u8=1; aclScalar* alpha1u8 = aclCreateScalar(&u8, ACL_UINT8);
  static int64_t i64=1; aclScalar* alpha1i64 = aclCreateScalar(&i64, ACL_INT64);
  static bool    bv=true; aclScalar* alpha1b = aclCreateScalar(&bv, ACL_BOOL);

  // FLOAT (AddWithCastCompute<float>)
  { TEST_CASE("tiling_float");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> a={1,2,3,4}, b={10,20,30,40}, o(4,0); std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    auto ret=RunAdd(tA,tB,alpha1f,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(float)");
    VERIFY(VerifyFloat(dO,{11,22,33,44},1e-5f,1e-5f),"float result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // FLOAT16 (AddWithCastCompute<half>)
  { TEST_CASE("tiling_fp16");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int16_t> a={0x3C00,0x4000,0x4200,0x4400}, b={0x3C00,0x3C00,0x3C00,0x3C00}, o(4,0); std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_FLOAT16,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT16,&tB); CreateAclTensor(o,sh,&dO,ACL_FLOAT16,&tO);
    auto ret=RunAdd(tA,tB,alpha1f,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(fp16)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // BF16 (AddWithCastCompute<half> bf16)
  { TEST_CASE("tiling_bf16");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int16_t> a={0x3F80,0x4000,0x4040,0x4080}, b={0x3F80,0x3F80,0x3F80,0x3F80}, o(4,0); std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_BF16,&tA); CreateAclTensor(b,sh,&dB,ACL_BF16,&tB); CreateAclTensor(o,sh,&dO,ACL_BF16,&tO);
    auto ret=RunAdd(tA,tB,alpha1f,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(bf16)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // INT32 (AddWithoutCastCompute<int32_t>)
  { TEST_CASE("tiling_int32");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int32_t> a={10,20,30,40}, b={1,2,3,4}, o(4,0); std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_INT32,&tA); CreateAclTensor(b,sh,&dB,ACL_INT32,&tB); CreateAclTensor(o,sh,&dO,ACL_INT32,&tO);
    auto ret=RunAdd(tA,tB,alpha1i,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(int32)");
    VERIFY(VerifyInt32(dO,{11,22,33,44}),"int32 result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // INT8 (AddWithoutCastCompute<int8_t>)
  { TEST_CASE("tiling_int8");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int8_t> a={1,2,3,4}, b={10,10,10,10}, o(4,0); std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_INT8,&tA); CreateAclTensor(b,sh,&dB,ACL_INT8,&tB); CreateAclTensor(o,sh,&dO,ACL_INT8,&tO);
    auto ret=RunAdd(tA,tB,alpha1i8,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(int8)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // UINT8 (AddWithoutCastCompute<uint8_t>)
  { TEST_CASE("tiling_uint8");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<uint8_t> a={1,2,3,4}, b={10,10,10,10}, o(4,0); std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_UINT8,&tA); CreateAclTensor(b,sh,&dB,ACL_UINT8,&tB); CreateAclTensor(o,sh,&dO,ACL_UINT8,&tO);
    auto ret=RunAdd(tA,tB,alpha1u8,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(uint8)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // INT64 (AddWithoutCastCompute<int64_t>)
  { TEST_CASE("tiling_int64");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int64_t> a={100,200,300,400}, b={1,2,3,4}, o(4,0LL); std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_INT64,&tA); CreateAclTensor(b,sh,&dB,ACL_INT64,&tB); CreateAclTensor(o,sh,&dO,ACL_INT64,&tO);
    auto ret=RunAdd(tA,tB,alpha1i64,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(int64)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // BOOL (AddBoolCompute)
  { TEST_CASE("tiling_bool");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<uint8_t> a={1,0,1,0}, b={0,1,1,0}, o(4,0); std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_BOOL,&tA); CreateAclTensor(b,sh,&dB,ACL_BOOL,&tB); CreateAclTensor(o,sh,&dO,ACL_BOOL,&tO);
    auto ret=RunAdd(tA,tB,alpha1b,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(bool)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // COMPLEX64 (AddWithoutCastCompute<int64_t>) - 可能走AiCpu
  { TEST_CASE("tiling_complex64");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> a={1,0,2,0,3,0,4,0}, b={1,0,1,0,1,0,1,0}, o(8,0); std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_COMPLEX64,&tA); CreateAclTensor(b,sh,&dB,ACL_COMPLEX64,&tB); CreateAclTensor(o,sh,&dO,ACL_COMPLEX64,&tO);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnAddGetWorkspaceSize(tA,tB,alpha1f,tO,&ws,&ex);
    EXPECT_SUCCESS(ret,"aclnnAdd(complex64)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // FP16 x FLOAT -> FLOAT (AddMixDtypeCompute<half,float>)
  { TEST_CASE("tiling_fp16_fp32_mix");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int16_t> a={0x3C00,0x4000,0x4200,0x4400}; std::vector<float> b={1,2,3,4}, o(4,0); std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_FLOAT16,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    auto ret=RunAdd(tA,tB,alpha1f,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(fp16xfp32)");
    VERIFY(VerifyFloat(dO,{2,4,6,8},1e-3f,1e-3f),"fp16xfp32 result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // FLOAT x FP16 -> FLOAT (AddMixDtypeCompute<float,half>)
  { TEST_CASE("tiling_fp32_fp16_mix");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> a={1,2,3,4}; std::vector<int16_t> b={0x3C00,0x4000,0x4200,0x4400}; std::vector<float> o(4,0); std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT16,&tB); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    auto ret=RunAdd(tA,tB,alpha1f,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(fp32xfp16)");
    VERIFY(VerifyFloat(dO,{2,4,6,8},1e-3f,1e-3f),"fp32xfp16 result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  aclDestroyScalar(alpha1f); aclDestroyScalar(alpha1i); aclDestroyScalar(alpha1i8);
  aclDestroyScalar(alpha1u8); aclDestroyScalar(alpha1i64); aclDestroyScalar(alpha1b);
}

static void TestAlphaPaths(aclrtStream stream) {
  std::vector<int64_t> sh={4};
  std::vector<float> a={1,2,3,4}, b={10,10,10,10}, o(4,0);

  // alpha=1 直接 Add 路径
  { TEST_CASE("alpha_eq_1");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float v=1.f; aclScalar* alpha=aclCreateScalar(&v,ACL_FLOAT);
    auto ret=RunAdd(tA,tB,alpha,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(alpha=1)");
    VERIFY(VerifyFloat(dO,{11,12,13,14},1e-5f,1e-5f),"alpha=1 result");
    aclDestroyScalar(alpha); aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // alpha=2 -> Axpy 路径 (模拟器不支持，覆盖参数校验路径)
  { TEST_CASE("alpha_eq_2_axpy_path");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float v=2.f; aclScalar* alpha=aclCreateScalar(&v,ACL_FLOAT);
    auto ret=RunAdd(tA,tB,alpha,tO,stream); EXPECT_FAIL(ret,"aclnnAdd(alpha=2,axpy)");
    aclDestroyScalar(alpha); aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // alpha=0 -> Axpy 路径
  { TEST_CASE("alpha_eq_0");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float v=0.f; aclScalar* alpha=aclCreateScalar(&v,ACL_FLOAT);
    auto ret=RunAdd(tA,tB,alpha,tO,stream); EXPECT_FAIL(ret,"aclnnAdd(alpha=0)");
    aclDestroyScalar(alpha); aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // alpha=-1
  { TEST_CASE("alpha_neg_1");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float v=-1.f; aclScalar* alpha=aclCreateScalar(&v,ACL_FLOAT);
    auto ret=RunAdd(tA,tB,alpha,tO,stream); EXPECT_FAIL(ret,"aclnnAdd(alpha=-1)");
    aclDestroyScalar(alpha); aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // int32 alpha=2 -> AxpyV2 路径
  { TEST_CASE("alpha_int32_axpyv2");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int32_t> ia={1,2,3,4}, ib={10,10,10,10}, io(4,0);
    CreateAclTensor(ia,sh,&dA,ACL_INT32,&tA); CreateAclTensor(ib,sh,&dB,ACL_INT32,&tB); CreateAclTensor(io,sh,&dO,ACL_INT32,&tO);
    static int32_t v=2; aclScalar* alpha=aclCreateScalar(&v,ACL_INT32);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnAddGetWorkspaceSize(tA,tB,alpha,tO,&ws,&ex);
    EXPECT_FAIL(ret,"aclnnAdd(int32,alpha=2,axpyv2)"); // AxpyV2 not supported in simulator
    aclDestroyScalar(alpha); aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }
}

static void TestShapeAndBoundary(aclrtStream stream) {
  static float f1=1.f; aclScalar* alpha1=aclCreateScalar(&f1,ACL_FLOAT);

  // 广播 [2,3] x [3]
  { TEST_CASE("broadcast_2x3_x_3");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> a={1,2,3,4,5,6}, b={10,20,30}, o(6,0);
    CreateAclTensor(a,{2,3},&dA,ACL_FLOAT,&tA); CreateAclTensor(b,{3},&dB,ACL_FLOAT,&tB); CreateAclTensor(o,{2,3},&dO,ACL_FLOAT,&tO);
    auto ret=RunAdd(tA,tB,alpha1,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(bcast 2x3 x 3)");
    VERIFY(VerifyFloat(dO,{11,22,33,14,25,36},1e-5f,1e-5f),"broadcast result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // 广播 [1] x [4,4]
  { TEST_CASE("broadcast_1_x_4x4");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> a={5.f}, b(16,1.f), o(16,0.f);
    CreateAclTensor(a,{1},&dA,ACL_FLOAT,&tA); CreateAclTensor(b,{4,4},&dB,ACL_FLOAT,&tB); CreateAclTensor(o,{4,4},&dO,ACL_FLOAT,&tO);
    auto ret=RunAdd(tA,tB,alpha1,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(bcast 1 x 4x4)");
    VERIFY(VerifyFloat(dO,std::vector<float>(16,6.f),1e-5f,1e-5f),"bcast result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // 空 tensor
  { TEST_CASE("empty_tensor");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> empty={};
    CreateAclTensor(empty,{0,4},&dA,ACL_FLOAT,&tA); CreateAclTensor(empty,{0,4},&dB,ACL_FLOAT,&tB); CreateAclTensor(empty,{0,4},&dO,ACL_FLOAT,&tO);
    auto ret=RunAdd(tA,tB,alpha1,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(empty)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // 大 tensor [64,64]
  { TEST_CASE("large_64x64");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> a(64*64,1.f), b(64*64,2.f), o(64*64,0.f);
    CreateAclTensor(a,{64,64},&dA,ACL_FLOAT,&tA); CreateAclTensor(b,{64,64},&dB,ACL_FLOAT,&tB); CreateAclTensor(o,{64,64},&dO,ACL_FLOAT,&tO);
    auto ret=RunAdd(tA,tB,alpha1,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(64x64)");
    VERIFY(VerifyFloat(dO,std::vector<float>(64*64,3.f),1e-5f,1e-5f),"64x64 result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // 边界值
  { TEST_CASE("boundary_values");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    float fmax=std::numeric_limits<float>::max();
    std::vector<float> a={0.f,-1.f,fmax,1.f}, b={0.f,-2.f,0.f,-1.f}, o(4,0.f);
    CreateAclTensor(a,{4},&dA,ACL_FLOAT,&tA); CreateAclTensor(b,{4},&dB,ACL_FLOAT,&tB); CreateAclTensor(o,{4},&dO,ACL_FLOAT,&tO);
    auto ret=RunAdd(tA,tB,alpha1,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(boundary)");
    VERIFY(VerifyFloat(dO,{0.f,-3.f,fmax,0.f},1e-4f,1e-4f),"boundary result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // NaN/Inf
  { TEST_CASE("nan_inf");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    float nan_v=std::numeric_limits<float>::quiet_NaN(), inf_v=std::numeric_limits<float>::infinity();
    std::vector<float> a={nan_v,inf_v,1.f,0.f}, b={1.f,1.f,nan_v,inf_v}, o(4,0.f);
    CreateAclTensor(a,{4},&dA,ACL_FLOAT,&tA); CreateAclTensor(b,{4},&dB,ACL_FLOAT,&tB); CreateAclTensor(o,{4},&dO,ACL_FLOAT,&tO);
    auto ret=RunAdd(tA,tB,alpha1,tO,stream); EXPECT_SUCCESS(ret,"aclnnAdd(nan_inf)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  aclDestroyScalar(alpha1);
}

static void TestApiVariants(aclrtStream stream) {
  std::vector<int64_t> sh={2,3};
  static float f1=1.f; aclScalar* alpha1=aclCreateScalar(&f1,ACL_FLOAT);
  static float f2=2.f; aclScalar* alpha2=aclCreateScalar(&f2,ACL_FLOAT);

  // aclnnAdds: tensor + alpha*scalar, alpha=1
  { TEST_CASE("adds_alpha1");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> a={1,2,3,4,5,6}, o(6,0);
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float sv=10.f; aclScalar* sc=aclCreateScalar(&sv,ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnAddsGetWorkspaceSize(tA,sc,alpha1,tO,&ws,&ex); EXPECT_SUCCESS(ret,"aclnnAddsGetWorkspaceSize(alpha=1)");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnAdds(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"aclnnAdds(alpha=1)");
      aclrtSynchronizeStream(stream);
      VERIFY(VerifyFloat(dO,{11,12,13,14,15,16},1e-5f,1e-5f),"adds alpha=1 result");
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyScalar(sc); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // aclnnAdds: alpha=2 (Axpy 路径，模拟器不支持)
  { TEST_CASE("adds_alpha2_axpy");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> a={1,2,3,4,5,6}, o(6,0);
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float sv=5.f; aclScalar* sc=aclCreateScalar(&sv,ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnAddsGetWorkspaceSize(tA,sc,alpha2,tO,&ws,&ex); EXPECT_FAIL(ret,"aclnnAdds(alpha=2)");
    aclDestroyScalar(sc); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // aclnnInplaceAdd
  { TEST_CASE("inplace_add");
    void *dA=nullptr,*dB=nullptr; aclTensor *tA=nullptr,*tB=nullptr;
    std::vector<float> self={1,2,3,4,5,6}, b={10,10,10,10,10,10};
    CreateAclTensor(self,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnInplaceAddGetWorkspaceSize(tA,tB,alpha1,&ws,&ex); EXPECT_SUCCESS(ret,"aclnnInplaceAddGetWorkspaceSize");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnInplaceAdd(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"aclnnInplaceAdd");
      aclrtSynchronizeStream(stream);
      VERIFY(VerifyFloat(dA,{11,12,13,14,15,16},1e-5f,1e-5f),"inplace_add result");
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyTensor(tA);aclDestroyTensor(tB); aclrtFree(dA);aclrtFree(dB); }

  // aclnnInplaceAdds
  { TEST_CASE("inplace_adds");
    void *dA=nullptr; aclTensor *tA=nullptr;
    std::vector<float> self={1,2,3,4,5,6};
    CreateAclTensor(self,sh,&dA,ACL_FLOAT,&tA);
    static float sv=10.f; aclScalar* sc=aclCreateScalar(&sv,ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnInplaceAddsGetWorkspaceSize(tA,sc,alpha1,&ws,&ex); EXPECT_SUCCESS(ret,"aclnnInplaceAddsGetWorkspaceSize");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnInplaceAdds(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"aclnnInplaceAdds");
      aclrtSynchronizeStream(stream);
      VERIFY(VerifyFloat(dA,{11,12,13,14,15,16},1e-5f,1e-5f),"inplace_adds result");
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyScalar(sc); aclDestroyTensor(tA); aclrtFree(dA); }

  // aclnnAddV3: scalar + alpha*tensor, alpha=1
  { TEST_CASE("addv3_alpha1");
    void *dB=nullptr,*dO=nullptr; aclTensor *tB=nullptr,*tO=nullptr;
    std::vector<float> b={1,2,3,4,5,6}, o(6,0);
    CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float sv=10.f; aclScalar* self_sc=aclCreateScalar(&sv,ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnAddV3GetWorkspaceSize(self_sc,tB,alpha1,tO,&ws,&ex); EXPECT_SUCCESS(ret,"aclnnAddV3GetWorkspaceSize(alpha=1)");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnAddV3(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"aclnnAddV3(alpha=1)");
      aclrtSynchronizeStream(stream);
      VERIFY(VerifyFloat(dO,{11,12,13,14,15,16},1e-5f,1e-5f),"addv3 alpha=1 result");
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyScalar(self_sc); aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dB);aclrtFree(dO); }

  // aclnnAddV3: alpha=2 -> Axpy 路径
  { TEST_CASE("addv3_alpha2_axpy");
    void *dB=nullptr,*dO=nullptr; aclTensor *tB=nullptr,*tO=nullptr;
    std::vector<float> b={1,2,3,4,5,6}, o(6,0);
    CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float sv=10.f; aclScalar* self_sc=aclCreateScalar(&sv,ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnAddV3GetWorkspaceSize(self_sc,tB,alpha2,tO,&ws,&ex); EXPECT_FAIL(ret,"aclnnAddV3(alpha=2)");
    aclDestroyScalar(self_sc); aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dB);aclrtFree(dO); }

  // aclnnInplaceAddV3
  { TEST_CASE("inplace_addv3");
    void *dB=nullptr; aclTensor *tB=nullptr;
    std::vector<float> b={1,2,3,4,5,6};
    CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB);
    static float sv=10.f; aclScalar* self_sc=aclCreateScalar(&sv,ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnInplaceAddV3GetWorkspaceSize(self_sc,tB,alpha1,&ws,&ex); EXPECT_SUCCESS(ret,"aclnnInplaceAddV3GetWorkspaceSize");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnInplaceAddV3(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"aclnnInplaceAddV3");
      aclrtSynchronizeStream(stream); if(wsp) aclrtFree(wsp);
    }
    aclDestroyScalar(self_sc); aclDestroyTensor(tB); aclrtFree(dB); }

  aclDestroyScalar(alpha1); aclDestroyScalar(alpha2);
}

static void TestErrorCases(aclrtStream stream) {
  std::vector<int64_t> sh={2,2}; std::vector<float> data={1,2,3,4};
  void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
  CreateAclTensor(data,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(data,sh,&dO,ACL_FLOAT,&tO);
  static float av=1.f; aclScalar* alpha=aclCreateScalar(&av,ACL_FLOAT);

  { TEST_CASE("error_nullptr_self"); uint64_t ws=0; aclOpExecutor* ex=nullptr;
    EXPECT_FAIL(aclnnAddGetWorkspaceSize(nullptr,tA,alpha,tO,&ws,&ex),"null self"); }
  { TEST_CASE("error_nullptr_other"); uint64_t ws=0; aclOpExecutor* ex=nullptr;
    EXPECT_FAIL(aclnnAddGetWorkspaceSize(tA,nullptr,alpha,tO,&ws,&ex),"null other"); }
  { TEST_CASE("error_nullptr_alpha"); uint64_t ws=0; aclOpExecutor* ex=nullptr;
    EXPECT_FAIL(aclnnAddGetWorkspaceSize(tA,tA,nullptr,tO,&ws,&ex),"null alpha"); }
  { TEST_CASE("error_nullptr_out"); uint64_t ws=0; aclOpExecutor* ex=nullptr;
    EXPECT_FAIL(aclnnAddGetWorkspaceSize(tA,tA,alpha,nullptr,&ws,&ex),"null out"); }
  { TEST_CASE("error_bad_shape");
    void *dB=nullptr,*dC=nullptr; aclTensor *tB=nullptr,*tC=nullptr;
    void *dA2=nullptr; aclTensor *tA2=nullptr;
    CreateAclTensor(std::vector<float>(6,1),{2,3},&dA2,ACL_FLOAT,&tA2);
    CreateAclTensor(std::vector<float>(8,1),{2,4},&dB,ACL_FLOAT,&tB);
    CreateAclTensor(std::vector<float>(6,0),{2,3},&dC,ACL_FLOAT,&tC);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    EXPECT_FAIL(aclnnAddGetWorkspaceSize(tA2,tB,alpha,tC,&ws,&ex),"bad shape");
    aclDestroyTensor(tA2);aclDestroyTensor(tB);aclDestroyTensor(tC);
    aclrtFree(dA2);aclrtFree(dB);aclrtFree(dC); }

  aclDestroyScalar(alpha); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO);
}

int main() {
  int32_t deviceId = 0; aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  TestTilingDtypes(stream);
  TestAlphaPaths(stream);
  TestShapeAndBoundary(stream);
  TestApiVariants(stream);
  TestErrorCases(stream);

  printf("\n========== RESULT: %d passed, %d failed ==========\n", g_passCount, g_failCount);
  aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize();
  return g_failCount > 0 ? 1 : 0;
}
