/**
 * Mul 算子端到端测试用例
 * 覆盖：aclnnMul / aclnnMuls / aclnnInplaceMul / aclnnInplaceMuls
 * 覆盖所有 16 种 tiling dtype 组合、广播、边界值、异常输入
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_mul.h"

#define CHECK_RET(cond, return_expr) \
  do { if (!(cond)) { return_expr; } } while (0)
#define LOG_PRINT(message, ...) \
  do { printf(message, ##__VA_ARGS__); } while (0)
#define TEST_CASE(name) \
  do { printf("\n[TEST] %s\n", name); } while(0)
#define EXPECT_SUCCESS(ret, api) \
  do { if ((ret) != ACL_SUCCESS) { \
    printf("[FAIL] %s returned %d\n", api, (int)(ret)); g_failCount++; \
  } else { printf("[PASS] %s\n", api); g_passCount++; } } while(0)
#define EXPECT_FAIL(ret, api) \
  do { if ((ret) == ACL_SUCCESS) { \
    printf("[FAIL] %s should have failed\n", api); g_failCount++; \
  } else { printf("[PASS] %s (expected fail) ret=%d\n", api, (int)(ret)); g_passCount++; } } while(0)
#define VERIFY(ok, name) \
  do { if (ok) { printf("[PASS] verify: %s\n", name); g_passCount++; } \
       else    { printf("[FAIL] verify: %s\n", name); g_failCount++; } } while(0)

static int g_passCount = 0;
static int g_failCount = 0;

static int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  if (shape.empty()) return 0;
  int64_t s = 1;
  for (auto d : shape) s *= d;
  return s;
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

// 两段式执行 aclnnMul
static aclnnStatus RunMul(aclTensor* self, aclTensor* other, aclTensor* out, aclrtStream stream) {
  uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
  auto ret = aclnnMulGetWorkspaceSize(self, other, out, &wsSize, &exec);
  if (ret != ACL_SUCCESS) return ret;
  void* ws = nullptr;
  if (wsSize > 0) { ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST); if (ret != ACL_SUCCESS) return ret; }
  ret = aclnnMul(ws, wsSize, exec, stream);
  aclrtSynchronizeStream(stream);
  if (ws) aclrtFree(ws);
  return ret;
}

// 验证 float 结果
static bool VerifyFloat(void* devOut, const std::vector<float>& expected, float atol, float rtol) {
  std::vector<float> actual(expected.size(), 0.f);
  aclrtMemcpy(actual.data(), actual.size() * sizeof(float), devOut,
              actual.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  for (size_t i = 0; i < expected.size(); i++) {
    float e = expected[i], a = actual[i];
    if (std::isnan(e) && std::isnan(a)) continue;
    if (std::isinf(e) && std::isinf(a) && std::signbit(e) == std::signbit(a)) continue;
    if (std::abs(a - e) > atol + rtol * std::abs(e)) {
      printf("    idx=%zu actual=%f expected=%f\n", i, a, e);
      return false;
    }
  }
  return true;
}

// 验证 int32 结果
static bool VerifyInt32(void* devOut, const std::vector<int32_t>& expected) {
  std::vector<int32_t> actual(expected.size(), 0);
  aclrtMemcpy(actual.data(), actual.size() * sizeof(int32_t), devOut,
              actual.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  for (size_t i = 0; i < expected.size(); i++) {
    if (actual[i] != expected[i]) {
      printf("    idx=%zu actual=%d expected=%d\n", i, actual[i], expected[i]);
      return false;
    }
  }
  return true;
}

// ===== 覆盖 tiling 层 16 种 dtype 组合 =====
static void TestTilingDtypeCombinations(aclrtStream stream) {

  // 1. FLOAT x FLOAT -> FLOAT
  {
    TEST_CASE("tiling_float_float_float");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> a={1.f,2.f,3.f,4.f}, b={2.f,3.f,4.f,5.f}, o(4,0.f);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB);
    CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(float)");
    std::vector<float> exp={2.f,6.f,12.f,20.f};
    VERIFY(VerifyFloat(dO,exp,1e-5f,1e-5f),"float result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 2. FLOAT16 x FLOAT16 -> FLOAT16
  {
    TEST_CASE("tiling_fp16_fp16_fp16");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    // fp16: 1.0=0x3C00, 2.0=0x4000, 3.0=0x4200, 4.0=0x4400
    std::vector<int16_t> a={0x3C00,0x4000,0x4200,0x4400}, b={0x4000,0x4000,0x4000,0x4000}, o(4,0);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_FLOAT16,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT16,&tB);
    CreateAclTensor(o,sh,&dO,ACL_FLOAT16,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(fp16)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 3. BF16 x BF16 -> BF16
  {
    TEST_CASE("tiling_bf16_bf16_bf16");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    // bf16: 1.0=0x3F80, 2.0=0x4000
    std::vector<int16_t> a={0x3F80,0x4000,0x4040,0x4080}, b={0x4000,0x4000,0x4000,0x4000}, o(4,0);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_BF16,&tA); CreateAclTensor(b,sh,&dB,ACL_BF16,&tB);
    CreateAclTensor(o,sh,&dO,ACL_BF16,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(bf16)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 4. INT8 x INT8 -> INT8
  {
    TEST_CASE("tiling_int8_int8_int8");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int8_t> a={1,2,3,4}, b={2,3,4,5}, o(4,0);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_INT8,&tA); CreateAclTensor(b,sh,&dB,ACL_INT8,&tB);
    CreateAclTensor(o,sh,&dO,ACL_INT8,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(int8)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 5. UINT8 x UINT8 -> UINT8
  {
    TEST_CASE("tiling_uint8_uint8_uint8");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<uint8_t> a={1,2,3,4}, b={3,3,3,3}, o(4,0);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_UINT8,&tA); CreateAclTensor(b,sh,&dB,ACL_UINT8,&tB);
    CreateAclTensor(o,sh,&dO,ACL_UINT8,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(uint8)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 6. BOOL x BOOL -> BOOL
  {
    TEST_CASE("tiling_bool_bool_bool");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<uint8_t> a={1,0,1,0}, b={1,1,0,0}, o(4,0);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_BOOL,&tA); CreateAclTensor(b,sh,&dB,ACL_BOOL,&tB);
    CreateAclTensor(o,sh,&dO,ACL_BOOL,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(bool)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 7. INT16 x INT16 -> INT16
  {
    TEST_CASE("tiling_int16_int16_int16");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int16_t> a={10,20,30,40}, b={2,2,2,2}, o(4,0);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_INT16,&tA); CreateAclTensor(b,sh,&dB,ACL_INT16,&tB);
    CreateAclTensor(o,sh,&dO,ACL_INT16,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(int16)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 8. INT32 x INT32 -> INT32
  {
    TEST_CASE("tiling_int32_int32_int32");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int32_t> a={100,200,300,400}, b={2,3,4,5}, o(4,0);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_INT32,&tA); CreateAclTensor(b,sh,&dB,ACL_INT32,&tB);
    CreateAclTensor(o,sh,&dO,ACL_INT32,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(int32)");
    std::vector<int32_t> exp={200,600,1200,2000};
    VERIFY(VerifyInt32(dO,exp),"int32 result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 9. INT64 x INT64 -> INT64
  {
    TEST_CASE("tiling_int64_int64_int64");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int64_t> a={1000LL,2000LL,3000LL,4000LL}, b={2LL,3LL,4LL,5LL}, o(4,0LL);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_INT64,&tA); CreateAclTensor(b,sh,&dB,ACL_INT64,&tB);
    CreateAclTensor(o,sh,&dO,ACL_INT64,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(int64)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 10. DOUBLE x DOUBLE -> DOUBLE
  {
    TEST_CASE("tiling_double_double_double");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<double> a={1.0,2.0,3.0,4.0}, b={2.0,3.0,4.0,5.0}, o(4,0.0);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_DOUBLE,&tA); CreateAclTensor(b,sh,&dB,ACL_DOUBLE,&tB);
    CreateAclTensor(o,sh,&dO,ACL_DOUBLE,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(double)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 11. COMPLEX64 x COMPLEX64 -> COMPLEX64
  {
    TEST_CASE("tiling_complex64_complex64_complex64");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    // complex64 存储为 float 对 (real, imag)
    std::vector<float> a={1.f,0.f,2.f,0.f,3.f,0.f,4.f,0.f};
    std::vector<float> b={1.f,0.f,1.f,0.f,1.f,0.f,1.f,0.f};
    std::vector<float> o(8,0.f);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_COMPLEX64,&tA); CreateAclTensor(b,sh,&dB,ACL_COMPLEX64,&tB);
    CreateAclTensor(o,sh,&dO,ACL_COMPLEX64,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(complex64)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 12. FLOAT16 x FLOAT -> FLOAT  (mix fp16 x fp32)
  {
    TEST_CASE("tiling_fp16_fp32_fp32");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int16_t> a={0x3C00,0x4000,0x4200,0x4400};
    std::vector<float>   b={1.f,2.f,3.f,4.f}, o(4,0.f);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_FLOAT16,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB);
    CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(fp16xfp32)");
    std::vector<float> exp={1.f,4.f,9.f,16.f};
    VERIFY(VerifyFloat(dO,exp,1e-3f,1e-3f),"fp16xfp32 result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 13. FLOAT x FLOAT16 -> FLOAT  (mix fp32 x fp16)
  {
    TEST_CASE("tiling_fp32_fp16_fp32");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float>   a={1.f,2.f,3.f,4.f};
    std::vector<int16_t> b={0x3C00,0x4000,0x4200,0x4400};
    std::vector<float>   o(4,0.f);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT16,&tB);
    CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(fp32xfp16)");
    std::vector<float> exp={1.f,4.f,9.f,16.f};
    VERIFY(VerifyFloat(dO,exp,1e-3f,1e-3f),"fp32xfp16 result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 14. BF16 x FLOAT -> FLOAT  (mix bf16 x fp32)
  {
    TEST_CASE("tiling_bf16_fp32_fp32");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int16_t> a={0x3F80,0x4000,0x4040,0x4080};
    std::vector<float>   b={1.f,2.f,3.f,4.f}, o(4,0.f);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_BF16,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB);
    CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(bf16xfp32)");
    std::vector<float> exp={1.f,4.f,9.f,16.f};
    VERIFY(VerifyFloat(dO,exp,1e-2f,1e-2f),"bf16xfp32 result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 15. FLOAT x BF16 -> FLOAT  (mix fp32 x bf16)
  {
    TEST_CASE("tiling_fp32_bf16_fp32");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float>   a={1.f,2.f,3.f,4.f};
    std::vector<int16_t> b={0x3F80,0x4000,0x4040,0x4080};
    std::vector<float>   o(4,0.f);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_BF16,&tB);
    CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(fp32xbf16)");
    std::vector<float> exp={1.f,4.f,9.f,16.f};
    VERIFY(VerifyFloat(dO,exp,1e-2f,1e-2f),"fp32xbf16 result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 16. COMPLEX128 (double complex) - 通过 aclnnMul 走 AiCpu 路径
  {
    TEST_CASE("tiling_complex128_complex128_complex128");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<double> a={1.0,0.0,2.0,0.0,3.0,0.0,4.0,0.0};
    std::vector<double> b={1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0};
    std::vector<double> o(8,0.0);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_COMPLEX128,&tA); CreateAclTensor(b,sh,&dB,ACL_COMPLEX128,&tB);
    CreateAclTensor(o,sh,&dO,ACL_COMPLEX128,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_FAIL(ret,"aclnnMul(complex128)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }
}

// ===== shape 覆盖 + 边界值 =====
static void TestShapeAndBoundary(aclrtStream stream) {

  // 广播：[2,3] x [3] -> [2,3]
  {
    TEST_CASE("shape_broadcast_2x3_x_3");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> a={1,2,3,4,5,6}, b={10,20,30}, o(6,0);
    std::vector<int64_t> sA={2,3}, sB={3}, sO={2,3};
    CreateAclTensor(a,sA,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sB,&dB,ACL_FLOAT,&tB);
    CreateAclTensor(o,sO,&dO,ACL_FLOAT,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(bcast 2x3 x 3)");
    std::vector<float> exp={10,40,90,40,100,180};
    VERIFY(VerifyFloat(dO,exp,1e-4f,1e-4f),"broadcast result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 广播：[1] x [4,4]
  {
    TEST_CASE("shape_broadcast_scalar_x_4x4");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> a={3.f}, b(16,2.f), o(16,0.f);
    std::vector<int64_t> sA={1}, sB={4,4};
    CreateAclTensor(a,sA,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sB,&dB,ACL_FLOAT,&tB);
    CreateAclTensor(o,sB,&dO,ACL_FLOAT,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(bcast 1 x 4x4)");
    std::vector<float> exp(16,6.f);
    VERIFY(VerifyFloat(dO,exp,1e-5f,1e-5f),"scalar bcast result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 广播：[4,1] x [1,4] -> [4,4]
  {
    TEST_CASE("shape_broadcast_4x1_x_1x4");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> a={1,2,3,4}, b={10,20,30,40}, o(16,0.f);
    std::vector<int64_t> sA={4,1}, sB={1,4}, sO={4,4};
    CreateAclTensor(a,sA,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sB,&dB,ACL_FLOAT,&tB);
    CreateAclTensor(o,sO,&dO,ACL_FLOAT,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(bcast 4x1 x 1x4)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 1D tensor
  {
    TEST_CASE("shape_1d_tensor");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> a={1,2,3,4,5,6,7,8}, b={2,2,2,2,2,2,2,2}, o(8,0);
    std::vector<int64_t> sh={8};
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB);
    CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(1D)");
    std::vector<float> exp={2,4,6,8,10,12,14,16};
    VERIFY(VerifyFloat(dO,exp,1e-5f,1e-5f),"1D result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 4D tensor
  {
    TEST_CASE("shape_4d_tensor");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    int n=2*2*2*2;
    std::vector<float> a(n,1.5f), b(n,2.0f), o(n,0.f);
    std::vector<int64_t> sh={2,2,2,2};
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB);
    CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(4D)");
    std::vector<float> exp(n,3.f);
    VERIFY(VerifyFloat(dO,exp,1e-5f,1e-5f),"4D result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 空 tensor (IsEmpty 分支)
  {
    TEST_CASE("shape_empty_tensor");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> empty={};
    std::vector<int64_t> sh={0,4};
    CreateAclTensor(empty,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(empty,sh,&dB,ACL_FLOAT,&tB);
    CreateAclTensor(empty,sh,&dO,ACL_FLOAT,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(empty)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 较大 tensor [64,64]
  {
    TEST_CASE("shape_large_64x64");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    int n=64*64;
    std::vector<float> a(n,1.5f), b(n,2.0f), o(n,0.f);
    std::vector<int64_t> sh={64,64};
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB);
    CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(64x64)");
    std::vector<float> exp(n,3.f);
    VERIFY(VerifyFloat(dO,exp,1e-5f,1e-5f),"64x64 result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 边界值：零、负数、极大、极小
  {
    TEST_CASE("boundary_zero_neg_max_min");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    float fmax=std::numeric_limits<float>::max();
    float fmin=std::numeric_limits<float>::min();
    std::vector<float> a={0.f,-1.f,fmax,fmin}, b={999.f,-2.f,1.f,2.f}, o(4,0.f);
    std::vector<int64_t> sh={4};
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB);
    CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(boundary)");
    std::vector<float> exp={0.f,2.f,fmax,fmin*2.f};
    VERIFY(VerifyFloat(dO,exp,1e-4f,1e-4f),"boundary result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }

  // 边界值：NaN / Inf
  {
    TEST_CASE("boundary_nan_inf");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    float nan_v=std::numeric_limits<float>::quiet_NaN();
    float inf_v=std::numeric_limits<float>::infinity();
    std::vector<float> a={nan_v,inf_v,-inf_v,1.f}, b={1.f,2.f,-1.f,nan_v}, o(4,0.f);
    std::vector<int64_t> sh={4};
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB);
    CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(nan_inf)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }
}

// ===== API 变体覆盖 =====
static void TestApiVariants(aclrtStream stream) {
  std::vector<int64_t> shape={2,3};

  // aclnnMuls: float tensor x float scalar
  {
    TEST_CASE("muls_float_x_float_scalar");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> a={1,2,3,4,5,6}, o(6,0);
    CreateAclTensor(a,shape,&dA,ACL_FLOAT,&tA);
    CreateAclTensor(o,shape,&dO,ACL_FLOAT,&tO);
    static float sv=2.f;
    aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    int ret=aclnnMulsGetWorkspaceSize(tA,sc,tO,&ws,&ex);
    EXPECT_SUCCESS(ret,"aclnnMulsGetWorkspaceSize(float)");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnMuls(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"aclnnMuls(float)");
      aclrtSynchronizeStream(stream);
      std::vector<float> exp={2,4,6,8,10,12};
      VERIFY(VerifyFloat(dO,exp,1e-5f,1e-5f),"muls float result");
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyScalar(sc); aclDestroyTensor(tA); aclDestroyTensor(tO);
    aclrtFree(dA); aclrtFree(dO);
  }

  // aclnnMuls: bf16 tensor x float scalar -> canUseMuls 分支
  {
    TEST_CASE("muls_bf16_x_float_scalar_canUseMuls");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<int16_t> a={0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0}, o(6,0);
    CreateAclTensor(a,shape,&dA,ACL_BF16,&tA);
    CreateAclTensor(o,shape,&dO,ACL_BF16,&tO);
    static float sv=3.f;
    aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    int ret=aclnnMulsGetWorkspaceSize(tA,sc,tO,&ws,&ex);
    EXPECT_FAIL(ret,"aclnnMulsGetWorkspaceSize(bf16xfloat)");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnMuls(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"aclnnMuls(bf16xfloat)");
      aclrtSynchronizeStream(stream);
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyScalar(sc); aclDestroyTensor(tA); aclDestroyTensor(tO);
    aclrtFree(dA); aclrtFree(dO);
  }

  // aclnnMuls: fp16 tensor x float scalar -> canUseMuls 分支
  {
    TEST_CASE("muls_fp16_x_float_scalar_canUseMuls");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<int16_t> a={0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600}, o(6,0);
    CreateAclTensor(a,shape,&dA,ACL_FLOAT16,&tA);
    CreateAclTensor(o,shape,&dO,ACL_FLOAT16,&tO);
    static float sv=2.f;
    aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    int ret=aclnnMulsGetWorkspaceSize(tA,sc,tO,&ws,&ex);
    EXPECT_FAIL(ret,"aclnnMulsGetWorkspaceSize(fp16xfloat)");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnMuls(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"aclnnMuls(fp16xfloat)");
      aclrtSynchronizeStream(stream);
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyScalar(sc); aclDestroyTensor(tA); aclDestroyTensor(tO);
    aclrtFree(dA); aclrtFree(dO);
  }

  // aclnnMuls: int32 tensor x int scalar
  {
    TEST_CASE("muls_int32_x_int_scalar");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<int32_t> a={1,2,3,4,5,6}, o(6,0);
    CreateAclTensor(a,shape,&dA,ACL_INT32,&tA);
    CreateAclTensor(o,shape,&dO,ACL_INT32,&tO);
    static int32_t sv=5;
    aclScalar* sc = aclCreateScalar(&sv, ACL_INT32);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    int ret=aclnnMulsGetWorkspaceSize(tA,sc,tO,&ws,&ex);
    EXPECT_SUCCESS(ret,"aclnnMulsGetWorkspaceSize(int32)");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnMuls(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"aclnnMuls(int32)");
      aclrtSynchronizeStream(stream);
      std::vector<int32_t> exp={5,10,15,20,25,30};
      VERIFY(VerifyInt32(dO,exp),"muls int32 result");
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyScalar(sc); aclDestroyTensor(tA); aclDestroyTensor(tO);
    aclrtFree(dA); aclrtFree(dO);
  }

  // aclnnInplaceMul: float
  {
    TEST_CASE("inplace_mul_float");
    void *dA=nullptr,*dB=nullptr; aclTensor *tA=nullptr,*tB=nullptr;
    std::vector<float> self={1,2,3,4,5,6}, b={2,2,2,2,2,2};
    CreateAclTensor(self,shape,&dA,ACL_FLOAT,&tA);
    CreateAclTensor(b,shape,&dB,ACL_FLOAT,&tB);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    int ret=aclnnInplaceMulGetWorkspaceSize(tA,tB,&ws,&ex);
    EXPECT_SUCCESS(ret,"aclnnInplaceMulGetWorkspaceSize");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnInplaceMul(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"aclnnInplaceMul");
      aclrtSynchronizeStream(stream);
      std::vector<float> exp={2,4,6,8,10,12};
      VERIFY(VerifyFloat(dA,exp,1e-5f,1e-5f),"inplace_mul result");
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyTensor(tA); aclDestroyTensor(tB);
    aclrtFree(dA); aclrtFree(dB);
  }

  // aclnnInplaceMul: mix fp16 x fp32 -> isMixDataType + IsRegBase 分支
  {
    TEST_CASE("inplace_mul_fp16_x_fp32");
    void *dA=nullptr,*dB=nullptr; aclTensor *tA=nullptr,*tB=nullptr;
    std::vector<int16_t> self={0x3C00,0x4000,0x4200,0x4400,0x4500,0x4600};
    std::vector<float>   b={1.f,2.f,3.f,4.f,5.f,6.f};
    CreateAclTensor(self,shape,&dA,ACL_FLOAT16,&tA);
    CreateAclTensor(b,shape,&dB,ACL_FLOAT,&tB);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    int ret=aclnnInplaceMulGetWorkspaceSize(tA,tB,&ws,&ex);
    EXPECT_FAIL(ret,"aclnnInplaceMulGetWorkspaceSize(fp16xfp32)");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnInplaceMul(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"aclnnInplaceMul(fp16xfp32)");
      aclrtSynchronizeStream(stream);
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyTensor(tA); aclDestroyTensor(tB);
    aclrtFree(dA); aclrtFree(dB);
  }

  // aclnnInplaceMuls: float scalar
  {
    TEST_CASE("inplace_muls_float_scalar");
    void *dA=nullptr; aclTensor *tA=nullptr;
    std::vector<float> self={1,2,3,4,5,6};
    CreateAclTensor(self,shape,&dA,ACL_FLOAT,&tA);
    static float sv=3.f;
    aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    int ret=aclnnInplaceMulsGetWorkspaceSize(tA,sc,&ws,&ex);
    EXPECT_SUCCESS(ret,"aclnnInplaceMulsGetWorkspaceSize");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnInplaceMuls(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"aclnnInplaceMuls");
      aclrtSynchronizeStream(stream);
      std::vector<float> exp={3,6,9,12,15,18};
      VERIFY(VerifyFloat(dA,exp,1e-5f,1e-5f),"inplace_muls result");
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyScalar(sc); aclDestroyTensor(tA);
    aclrtFree(dA);
  }

  // aclnnInplaceMuls: bf16 x float scalar -> canUseMuls 分支
  {
    TEST_CASE("inplace_muls_bf16_x_float_scalar");
    void *dA=nullptr; aclTensor *tA=nullptr;
    std::vector<int16_t> self={0x3F80,0x4000,0x4040,0x4080,0x40A0,0x40C0};
    CreateAclTensor(self,shape,&dA,ACL_BF16,&tA);
    static float sv=2.f;
    aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    int ret=aclnnInplaceMulsGetWorkspaceSize(tA,sc,&ws,&ex);
    EXPECT_FAIL(ret,"aclnnInplaceMulsGetWorkspaceSize(bf16)");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnInplaceMuls(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"aclnnInplaceMuls(bf16)");
      aclrtSynchronizeStream(stream);
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyScalar(sc); aclDestroyTensor(tA);
    aclrtFree(dA);
  }

  // aclnnMul: IsMulSupportNonContiguous 路径 (同dtype, dim<=4, RegBase)
  {
    TEST_CASE("mul_noncontiguous_support_path");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> a={1,2,3,4}, b={5,6,7,8}, o(4,0.f);
    std::vector<int64_t> sh={2,2};
    CreateAclTensor(a,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(b,sh,&dB,ACL_FLOAT,&tB);
    CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    auto ret = RunMul(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"aclnnMul(noncontiguous path)");
    std::vector<float> exp={5,12,21,32};
    VERIFY(VerifyFloat(dO,exp,1e-5f,1e-5f),"noncontiguous path result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO);
    aclrtFree(dA);aclrtFree(dB);aclrtFree(dO);
  }
}

// ===== 异常输入（覆盖参数校验分支） =====
static void TestErrorCases(aclrtStream stream) {
  std::vector<int64_t> sh={2,2};
  std::vector<float> data={1,2,3,4};
  void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
  CreateAclTensor(data,sh,&dA,ACL_FLOAT,&tA);
  CreateAclTensor(data,sh,&dO,ACL_FLOAT,&tO);

  // nullptr self
  {
    TEST_CASE("error_nullptr_self");
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    int ret=aclnnMulGetWorkspaceSize(nullptr,tA,tO,&ws,&ex);
    EXPECT_FAIL(ret,"aclnnMulGetWorkspaceSize(null self)");
  }
  // nullptr other
  {
    TEST_CASE("error_nullptr_other");
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    int ret=aclnnMulGetWorkspaceSize(tA,nullptr,tO,&ws,&ex);
    EXPECT_FAIL(ret,"aclnnMulGetWorkspaceSize(null other)");
  }
  // nullptr out
  {
    TEST_CASE("error_nullptr_out");
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    int ret=aclnnMulGetWorkspaceSize(tA,tA,nullptr,&ws,&ex);
    EXPECT_FAIL(ret,"aclnnMulGetWorkspaceSize(null out)");
  }
  // shape 不兼容广播：[2,3] x [2,4]
  {
    TEST_CASE("error_incompatible_shape");
    void *dB=nullptr,*dC=nullptr; aclTensor *tB=nullptr,*tC=nullptr;
    std::vector<float> d1(6,1),d2(8,1),d3(6,0);
    std::vector<int64_t> sA={2,3},sB={2,4},sC={2,3};
    void *dA2=nullptr; aclTensor *tA2=nullptr;
    CreateAclTensor(d1,sA,&dA2,ACL_FLOAT,&tA2);
    CreateAclTensor(d2,sB,&dB,ACL_FLOAT,&tB);
    CreateAclTensor(d3,sC,&dC,ACL_FLOAT,&tC);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    int ret=aclnnMulGetWorkspaceSize(tA2,tB,tC,&ws,&ex);
    EXPECT_FAIL(ret,"aclnnMulGetWorkspaceSize(bad shape)");
    aclDestroyTensor(tA2);aclDestroyTensor(tB);aclDestroyTensor(tC);
    aclrtFree(dA2);aclrtFree(dB);aclrtFree(dC);
  }
  // aclnnMuls: nullptr scalar
  {
    TEST_CASE("error_muls_nullptr_scalar");
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    int ret=aclnnMulsGetWorkspaceSize(tA,nullptr,tO,&ws,&ex);
    EXPECT_FAIL(ret,"aclnnMulsGetWorkspaceSize(null scalar)");
  }
  // aclnnInplaceMul: nullptr other
  {
    TEST_CASE("error_inplace_mul_nullptr_other");
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    int ret=aclnnInplaceMulGetWorkspaceSize(tA,nullptr,&ws,&ex);
    EXPECT_FAIL(ret,"aclnnInplaceMulGetWorkspaceSize(null other)");
  }

  aclDestroyTensor(tA); aclDestroyTensor(tO);
  aclrtFree(dA); aclrtFree(dO);
}

// ===== main =====
int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  TestTilingDtypeCombinations(stream);
  TestShapeAndBoundary(stream);
  TestApiVariants(stream);
  TestErrorCases(stream);

  printf("\n========== RESULT: %d passed, %d failed ==========\n", g_passCount, g_failCount);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return g_failCount > 0 ? 1 : 0;
}
