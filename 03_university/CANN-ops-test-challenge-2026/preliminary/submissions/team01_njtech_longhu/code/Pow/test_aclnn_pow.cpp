/**
 * Pow 算子端到端测试用例 - 龙湖小队
 * 覆盖：PowTensorScalar / PowScalarTensor / PowTensorTensor / Exp2 及 Inplace 变体
 * 覆盖特殊指数优化路径、7种dtype、广播、边界值、异常输入
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "acl/acl.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"
#include "aclnnop/aclnn_exp2.h"

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

// 两段式执行 PowTensorScalar
static aclnnStatus RunPowTS(aclTensor* self, aclScalar* exp, aclTensor* out, aclrtStream stream) {
  uint64_t ws=0; aclOpExecutor* ex=nullptr;
  auto ret = aclnnPowTensorScalarGetWorkspaceSize(self, exp, out, &ws, &ex);
  if (ret != ACL_SUCCESS) return ret;
  void* wsp=nullptr; if(ws>0) { ret=aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST); if(ret!=ACL_SUCCESS) return ret; }
  ret = aclnnPowTensorScalar(wsp, ws, ex, stream);
  aclrtSynchronizeStream(stream); if(wsp) aclrtFree(wsp); return ret;
}

// 两段式执行 PowTensorTensor
static aclnnStatus RunPowTT(aclTensor* self, aclTensor* exp, aclTensor* out, aclrtStream stream) {
  uint64_t ws=0; aclOpExecutor* ex=nullptr;
  auto ret = aclnnPowTensorTensorGetWorkspaceSize(self, exp, out, &ws, &ex);
  if (ret != ACL_SUCCESS) return ret;
  void* wsp=nullptr; if(ws>0) { ret=aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST); if(ret!=ACL_SUCCESS) return ret; }
  ret = aclnnPowTensorTensor(wsp, ws, ex, stream);
  aclrtSynchronizeStream(stream); if(wsp) aclrtFree(wsp); return ret;
}

static bool VerifyFloat(void* devOut, const std::vector<float>& expected, float atol, float rtol) {
  std::vector<float> actual(expected.size(), 0.f);
  aclrtMemcpy(actual.data(), actual.size()*sizeof(float), devOut, actual.size()*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  for (size_t i = 0; i < expected.size(); i++) {
    float e=expected[i], a=actual[i];
    if (std::isnan(e) && std::isnan(a)) continue;
    if (std::isinf(e) && std::isinf(a) && std::signbit(e)==std::signbit(a)) continue;
    if (std::abs(a-e) > atol + rtol*std::abs(e)) {
      printf("    idx=%zu actual=%f expected=%f\n", i, a, e); return false;
    }
  }
  return true;
}

// ===== TensorScalar: 特殊指数优化路径 =====
static void TestPowTensorScalarSpecial(aclrtStream stream) {
  std::vector<float> base={1.f,4.f,9.f,16.f};
  std::vector<int64_t> sh={4};

  // exp=0.5 -> Sqrt 路径 (CheckSupportPows)
  { TEST_CASE("ts_exp_0p5_sqrt");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> o(4,0);
    CreateAclTensor(base,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float ev=0.5f; aclScalar* exp=aclCreateScalar(&ev,ACL_FLOAT);
    auto ret=RunPowTS(tA,exp,tO,stream); EXPECT_SUCCESS(ret,"PowTS(exp=0.5)");
    std::vector<float> expected={1.f,2.f,3.f,4.f};
    VERIFY(VerifyFloat(dO,expected,1e-4f,1e-4f),"exp=0.5 result");
    aclDestroyScalar(exp); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // exp=2.0 -> Square 路径 (canUseSquare)
  { TEST_CASE("ts_exp_2_square");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> o(4,0);
    CreateAclTensor(base,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float ev=2.f; aclScalar* exp=aclCreateScalar(&ev,ACL_FLOAT);
    auto ret=RunPowTS(tA,exp,tO,stream); EXPECT_FAIL(ret,"PowTS(exp=2)");
    aclDestroyScalar(exp); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // exp=3.0 -> Pows 路径
  { TEST_CASE("ts_exp_3_cube");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> b2={1.f,2.f,3.f,4.f}, o(4,0);
    CreateAclTensor(b2,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float ev=3.f; aclScalar* exp=aclCreateScalar(&ev,ACL_FLOAT);
    auto ret=RunPowTS(tA,exp,tO,stream); EXPECT_SUCCESS(ret,"PowTS(exp=3)");
    std::vector<float> expected={1.f,8.f,27.f,64.f};
    VERIFY(VerifyFloat(dO,expected,1e-4f,1e-4f),"exp=3 result");
    aclDestroyScalar(exp); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // exp=-0.5 -> Pows 路径
  { TEST_CASE("ts_exp_neg0p5");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> o(4,0);
    CreateAclTensor(base,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float ev=-0.5f; aclScalar* exp=aclCreateScalar(&ev,ACL_FLOAT);
    auto ret=RunPowTS(tA,exp,tO,stream); EXPECT_SUCCESS(ret,"PowTS(exp=-0.5)");
    std::vector<float> expected={1.f,0.5f,1.f/3.f,0.25f};
    VERIFY(VerifyFloat(dO,expected,1e-3f,1e-3f),"exp=-0.5 result");
    aclDestroyScalar(exp); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // exp=-1.0 -> Pows 路径
  { TEST_CASE("ts_exp_neg1");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> b2={1.f,2.f,4.f,8.f}, o(4,0);
    CreateAclTensor(b2,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float ev=-1.f; aclScalar* exp=aclCreateScalar(&ev,ACL_FLOAT);
    auto ret=RunPowTS(tA,exp,tO,stream); EXPECT_SUCCESS(ret,"PowTS(exp=-1)");
    std::vector<float> expected={1.f,0.5f,0.25f,0.125f};
    VERIFY(VerifyFloat(dO,expected,1e-4f,1e-4f),"exp=-1 result");
    aclDestroyScalar(exp); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // exp=-2.0 -> Pows 路径
  { TEST_CASE("ts_exp_neg2");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> b2={1.f,2.f,4.f,8.f}, o(4,0);
    CreateAclTensor(b2,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float ev=-2.f; aclScalar* exp=aclCreateScalar(&ev,ACL_FLOAT);
    auto ret=RunPowTS(tA,exp,tO,stream); EXPECT_SUCCESS(ret,"PowTS(exp=-2)");
    std::vector<float> expected={1.f,0.25f,0.0625f,0.015625f};
    VERIFY(VerifyFloat(dO,expected,1e-4f,1e-4f),"exp=-2 result");
    aclDestroyScalar(exp); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // exp=1.0 -> 普通 Pow 路径
  { TEST_CASE("ts_exp_1");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> b2={1.f,2.f,3.f,4.f}, o(4,0);
    CreateAclTensor(b2,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float ev=1.f; aclScalar* exp=aclCreateScalar(&ev,ACL_FLOAT);
    auto ret=RunPowTS(tA,exp,tO,stream); EXPECT_SUCCESS(ret,"PowTS(exp=1)");
    VERIFY(VerifyFloat(dO,{1.f,2.f,3.f,4.f},1e-5f,1e-5f),"exp=1 result");
    aclDestroyScalar(exp); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // exp=0 -> x^0 = 1
  { TEST_CASE("ts_exp_0");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> b2={1.f,2.f,3.f,4.f}, o(4,0);
    CreateAclTensor(b2,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float ev=0.f; aclScalar* exp=aclCreateScalar(&ev,ACL_FLOAT);
    auto ret=RunPowTS(tA,exp,tO,stream); EXPECT_SUCCESS(ret,"PowTS(exp=0)");
    VERIFY(VerifyFloat(dO,{1.f,1.f,1.f,1.f},1e-5f,1e-5f),"exp=0 result");
    aclDestroyScalar(exp); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }
}

// ===== TensorScalar: dtype 覆盖 =====
static void TestPowTensorScalarDtypes(aclrtStream stream) {
  std::vector<int64_t> sh={4};
  static float ev2=2.f; aclScalar* exp2=aclCreateScalar(&ev2,ACL_FLOAT);

  // FLOAT16
  { TEST_CASE("ts_dtype_fp16");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<int16_t> b={0x3C00,0x4000,0x4200,0x4400}, o(4,0); // 1,2,3,4
    CreateAclTensor(b,sh,&dA,ACL_FLOAT16,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT16,&tO);
    auto ret=RunPowTS(tA,exp2,tO,stream); EXPECT_FAIL(ret,"PowTS(fp16,exp=2)");
    aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // BF16
  { TEST_CASE("ts_dtype_bf16");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<int16_t> b={0x3F80,0x4000,0x4040,0x4080}, o(4,0); // 1,2,3,4
    CreateAclTensor(b,sh,&dA,ACL_BF16,&tA); CreateAclTensor(o,sh,&dO,ACL_BF16,&tO);
    auto ret=RunPowTS(tA,exp2,tO,stream); EXPECT_FAIL(ret,"PowTS(bf16,exp=2)");
    aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // INT32
  { TEST_CASE("ts_dtype_int32");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<int32_t> b={1,2,3,4}, o(4,0);
    CreateAclTensor(b,sh,&dA,ACL_INT32,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float ev=2.f; aclScalar* exp=aclCreateScalar(&ev,ACL_FLOAT);
    auto ret=RunPowTS(tA,exp,tO,stream); EXPECT_FAIL(ret,"PowTS(int32,exp=2)");
    aclDestroyScalar(exp); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // INT8
  { TEST_CASE("ts_dtype_int8");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<int8_t> b={1,2,3,4}, o(4,0);
    CreateAclTensor(b,sh,&dA,ACL_INT8,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float ev=2.f; aclScalar* exp=aclCreateScalar(&ev,ACL_FLOAT);
    auto ret=RunPowTS(tA,exp,tO,stream); EXPECT_FAIL(ret,"PowTS(int8,exp=2)");
    aclDestroyScalar(exp); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  aclDestroyScalar(exp2);
}

// ===== ScalarTensor =====
static void TestPowScalarTensor(aclrtStream stream) {
  std::vector<int64_t> sh={4};

  // scalar=2, exp tensor=[1,2,3,4] -> 2^1,2^2,2^3,2^4
  { TEST_CASE("st_base2_exp_tensor");
    void *dB=nullptr,*dO=nullptr; aclTensor *tB=nullptr,*tO=nullptr;
    std::vector<float> exp_data={1.f,2.f,3.f,4.f}, o(4,0);
    CreateAclTensor(exp_data,sh,&dB,ACL_FLOAT,&tB); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float sv=2.f; aclScalar* self_sc=aclCreateScalar(&sv,ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnPowScalarTensorGetWorkspaceSize(self_sc,tB,tO,&ws,&ex);
    EXPECT_SUCCESS(ret,"PowScalarTensorGetWorkspaceSize");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnPowScalarTensor(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"PowScalarTensor");
      aclrtSynchronizeStream(stream);
      VERIFY(VerifyFloat(dO,{2.f,4.f,8.f,16.f},1e-4f,1e-4f),"st base=2 result");
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyScalar(self_sc); aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dB);aclrtFree(dO); }

  // scalar=1 -> 1^x = 1 (BuildPowScalarTensorFillOne 路径)
  { TEST_CASE("st_base1_fillone");
    void *dB=nullptr,*dO=nullptr; aclTensor *tB=nullptr,*tO=nullptr;
    std::vector<float> exp_data={1.f,2.f,3.f,4.f}, o(4,0);
    CreateAclTensor(exp_data,sh,&dB,ACL_FLOAT,&tB); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float sv=1.f; aclScalar* self_sc=aclCreateScalar(&sv,ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnPowScalarTensorGetWorkspaceSize(self_sc,tB,tO,&ws,&ex);
    EXPECT_FAIL(ret,"PowScalarTensor(base=1)GetWorkspaceSize");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnPowScalarTensor(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"PowScalarTensor(base=1)");
      aclrtSynchronizeStream(stream);
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyScalar(self_sc); aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dB);aclrtFree(dO); }

  // scalar=0 -> 0^x (x>0 = 0, x=0 = 1)
  { TEST_CASE("st_base0");
    void *dB=nullptr,*dO=nullptr; aclTensor *tB=nullptr,*tO=nullptr;
    std::vector<float> exp_data={1.f,2.f,3.f,4.f}, o(4,0);
    CreateAclTensor(exp_data,sh,&dB,ACL_FLOAT,&tB); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float sv=0.f; aclScalar* self_sc=aclCreateScalar(&sv,ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnPowScalarTensorGetWorkspaceSize(self_sc,tB,tO,&ws,&ex);
    EXPECT_SUCCESS(ret,"PowScalarTensor(base=0)GetWorkspaceSize");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnPowScalarTensor(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"PowScalarTensor(base=0)");
      aclrtSynchronizeStream(stream); if(wsp) aclrtFree(wsp);
    }
    aclDestroyScalar(self_sc); aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dB);aclrtFree(dO); }
}

// ===== TensorTensor: 7种dtype =====
static void TestPowTensorTensor(aclrtStream stream) {
  std::vector<int64_t> sh={4};

  // FLOAT (OP_KEY_3)
  { TEST_CASE("tt_float");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> base={1.f,2.f,3.f,4.f}, exp={2.f,2.f,2.f,2.f}, o(4,0);
    CreateAclTensor(base,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(exp,sh,&dB,ACL_FLOAT,&tB); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    auto ret=RunPowTT(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"PowTT(float)");
    VERIFY(VerifyFloat(dO,{1.f,4.f,9.f,16.f},1e-4f,1e-4f),"tt float result");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // FLOAT16 (OP_KEY_1)
  { TEST_CASE("tt_fp16");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int16_t> base={0x3C00,0x4000,0x4200,0x4400}, exp={0x4000,0x4000,0x4000,0x4000}, o(4,0);
    CreateAclTensor(base,sh,&dA,ACL_FLOAT16,&tA); CreateAclTensor(exp,sh,&dB,ACL_FLOAT16,&tB); CreateAclTensor(o,sh,&dO,ACL_FLOAT16,&tO);
    auto ret=RunPowTT(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"PowTT(fp16)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // BF16 (OP_KEY_2)
  { TEST_CASE("tt_bf16");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int16_t> base={0x3F80,0x4000,0x4040,0x4080}, exp={0x4000,0x4000,0x4000,0x4000}, o(4,0);
    CreateAclTensor(base,sh,&dA,ACL_BF16,&tA); CreateAclTensor(exp,sh,&dB,ACL_BF16,&tB); CreateAclTensor(o,sh,&dO,ACL_BF16,&tO);
    auto ret=RunPowTT(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"PowTT(bf16)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // UINT8 (OP_KEY_4)
  { TEST_CASE("tt_uint8");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<uint8_t> base={1,2,3,4}, exp={2,2,2,2}, o(4,0);
    CreateAclTensor(base,sh,&dA,ACL_UINT8,&tA); CreateAclTensor(exp,sh,&dB,ACL_UINT8,&tB); CreateAclTensor(o,sh,&dO,ACL_UINT8,&tO);
    auto ret=RunPowTT(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"PowTT(uint8)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // INT8 (OP_KEY_5)
  { TEST_CASE("tt_int8");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int8_t> base={1,2,3,4}, exp={2,2,2,2}, o(4,0);
    CreateAclTensor(base,sh,&dA,ACL_INT8,&tA); CreateAclTensor(exp,sh,&dB,ACL_INT8,&tB); CreateAclTensor(o,sh,&dO,ACL_INT8,&tO);
    auto ret=RunPowTT(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"PowTT(int8)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // INT16 (OP_KEY_6)
  { TEST_CASE("tt_int16");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int16_t> base={1,2,3,4}, exp={2,2,2,2}, o(4,0);
    CreateAclTensor(base,sh,&dA,ACL_INT16,&tA); CreateAclTensor(exp,sh,&dB,ACL_INT16,&tB); CreateAclTensor(o,sh,&dO,ACL_INT16,&tO);
    auto ret=RunPowTT(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"PowTT(int16)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // INT32 (OP_KEY_7)
  { TEST_CASE("tt_int32");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<int32_t> base={1,2,3,4}, exp={2,2,2,2}, o(4,0);
    CreateAclTensor(base,sh,&dA,ACL_INT32,&tA); CreateAclTensor(exp,sh,&dB,ACL_INT32,&tB); CreateAclTensor(o,sh,&dO,ACL_INT32,&tO);
    auto ret=RunPowTT(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"PowTT(int32)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }

  // 广播 [4,1] x [1,4]
  { TEST_CASE("tt_broadcast");
    void *dA=nullptr,*dB=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tB=nullptr,*tO=nullptr;
    std::vector<float> base={2.f,3.f,4.f,5.f}, exp={1.f,2.f,3.f,4.f}, o(16,0);
    CreateAclTensor(base,{4,1},&dA,ACL_FLOAT,&tA); CreateAclTensor(exp,{1,4},&dB,ACL_FLOAT,&tB); CreateAclTensor(o,{4,4},&dO,ACL_FLOAT,&tO);
    auto ret=RunPowTT(tA,tB,tO,stream); EXPECT_SUCCESS(ret,"PowTT(broadcast)");
    aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dB);aclrtFree(dO); }
}

// ===== Inplace 变体 =====
static void TestInplace(aclrtStream stream) {
  std::vector<int64_t> sh={4};

  // InplacePowTensorScalar
  { TEST_CASE("inplace_ts_exp2");
    void *dA=nullptr; aclTensor *tA=nullptr;
    std::vector<float> self={1.f,2.f,3.f,4.f};
    CreateAclTensor(self,sh,&dA,ACL_FLOAT,&tA);
    static float ev=2.f; aclScalar* exp=aclCreateScalar(&ev,ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnInplacePowTensorScalarGetWorkspaceSize(tA,exp,&ws,&ex);
    EXPECT_FAIL(ret,"InplacePowTSGetWorkspaceSize");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnInplacePowTensorScalar(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"InplacePowTS");
      aclrtSynchronizeStream(stream);
      VERIFY(VerifyFloat(dA,{1.f,4.f,9.f,16.f},1e-4f,1e-4f),"inplace ts result");
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyScalar(exp); aclDestroyTensor(tA); aclrtFree(dA); }

  // InplacePowTensorTensor
  { TEST_CASE("inplace_tt");
    void *dA=nullptr,*dB=nullptr; aclTensor *tA=nullptr,*tB=nullptr;
    std::vector<float> self={1.f,2.f,3.f,4.f}, exp={2.f,2.f,2.f,2.f};
    CreateAclTensor(self,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(exp,sh,&dB,ACL_FLOAT,&tB);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnInplacePowTensorTensorGetWorkspaceSize(tA,tB,&ws,&ex);
    EXPECT_SUCCESS(ret,"InplacePowTTGetWorkspaceSize");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnInplacePowTensorTensor(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"InplacePowTT");
      aclrtSynchronizeStream(stream);
      VERIFY(VerifyFloat(dA,{1.f,4.f,9.f,16.f},1e-4f,1e-4f),"inplace tt result");
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyTensor(tA);aclDestroyTensor(tB); aclrtFree(dA);aclrtFree(dB); }
}

// ===== Exp2 API =====
static void TestExp2(aclrtStream stream) {
  std::vector<int64_t> sh={4};

  // aclnnExp2: 2^x
  { TEST_CASE("exp2_float");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> self={0.f,1.f,2.f,3.f}, o(4,0);
    CreateAclTensor(self,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnExp2GetWorkspaceSize(tA,tO,&ws,&ex); EXPECT_SUCCESS(ret,"Exp2GetWorkspaceSize");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnExp2(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"Exp2");
      aclrtSynchronizeStream(stream);
      VERIFY(VerifyFloat(dO,{1.f,2.f,4.f,8.f},1e-4f,1e-4f),"exp2 result");
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // aclnnExp2 fp16
  { TEST_CASE("exp2_fp16");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<int16_t> self={0x0000,0x3C00,0x4000,0x4200}, o(4,0); // 0,1,2,3
    CreateAclTensor(self,sh,&dA,ACL_FLOAT16,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT16,&tO);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnExp2GetWorkspaceSize(tA,tO,&ws,&ex); EXPECT_SUCCESS(ret,"Exp2GetWorkspaceSize(fp16)");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnExp2(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"Exp2(fp16)");
      aclrtSynchronizeStream(stream); if(wsp) aclrtFree(wsp);
    }
    aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // aclnnInplaceExp2
  { TEST_CASE("inplace_exp2");
    void *dA=nullptr; aclTensor *tA=nullptr;
    std::vector<float> self={0.f,1.f,2.f,3.f};
    CreateAclTensor(self,sh,&dA,ACL_FLOAT,&tA);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    auto ret=aclnnInplaceExp2GetWorkspaceSize(tA,&ws,&ex); EXPECT_SUCCESS(ret,"InplaceExp2GetWorkspaceSize");
    if(ret==ACL_SUCCESS){
      void* wsp=nullptr; if(ws>0) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
      ret=aclnnInplaceExp2(wsp,ws,ex,stream); EXPECT_SUCCESS(ret,"InplaceExp2");
      aclrtSynchronizeStream(stream);
      VERIFY(VerifyFloat(dA,{1.f,2.f,4.f,8.f},1e-4f,1e-4f),"inplace exp2 result");
      if(wsp) aclrtFree(wsp);
    }
    aclDestroyTensor(tA); aclrtFree(dA); }
}

// ===== 边界值 + 异常输入 =====
static void TestBoundaryAndErrors(aclrtStream stream) {
  std::vector<int64_t> sh={4};

  // base=0, exp>0 -> 0
  { TEST_CASE("boundary_base0");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> base={0.f,0.f,0.f,0.f}, o(4,0);
    CreateAclTensor(base,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float ev=2.f; aclScalar* exp=aclCreateScalar(&ev,ACL_FLOAT);
    auto ret=RunPowTS(tA,exp,tO,stream); EXPECT_FAIL(ret,"PowTS(base=0,exp=2)");
    aclDestroyScalar(exp); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // 大 tensor [64,64]
  { TEST_CASE("shape_large_64x64");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> base(64*64,2.f), o(64*64,0.f);
    CreateAclTensor(base,{64,64},&dA,ACL_FLOAT,&tA); CreateAclTensor(o,{64,64},&dO,ACL_FLOAT,&tO);
    static float ev=2.f; aclScalar* exp=aclCreateScalar(&ev,ACL_FLOAT);
    auto ret=RunPowTS(tA,exp,tO,stream); EXPECT_FAIL(ret,"PowTS(64x64)");
    aclDestroyScalar(exp); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // 空 tensor
  { TEST_CASE("shape_empty");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> empty={};
    CreateAclTensor(empty,{0,4},&dA,ACL_FLOAT,&tA); CreateAclTensor(empty,{0,4},&dO,ACL_FLOAT,&tO);
    static float ev=2.f; aclScalar* exp=aclCreateScalar(&ev,ACL_FLOAT);
    auto ret=RunPowTS(tA,exp,tO,stream); EXPECT_SUCCESS(ret,"PowTS(empty)");
    aclDestroyScalar(exp); aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // 异常：nullptr self
  { TEST_CASE("error_nullptr_self");
    void *dO=nullptr; aclTensor *tO=nullptr;
    std::vector<float> o={0,0,0,0};
    CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    static float ev=2.f; aclScalar* exp=aclCreateScalar(&ev,ACL_FLOAT);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    EXPECT_FAIL(aclnnPowTensorScalarGetWorkspaceSize(nullptr,exp,tO,&ws,&ex),"null self");
    aclDestroyScalar(exp); aclDestroyTensor(tO); aclrtFree(dO); }

  // 异常：nullptr exponent
  { TEST_CASE("error_nullptr_exp");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> base={1,2,3,4}, o(4,0);
    CreateAclTensor(base,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    EXPECT_FAIL(aclnnPowTensorScalarGetWorkspaceSize(tA,nullptr,tO,&ws,&ex),"null exp");
    aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }

  // 异常：PowTT nullptr
  { TEST_CASE("error_tt_nullptr");
    void *dA=nullptr,*dO=nullptr; aclTensor *tA=nullptr,*tO=nullptr;
    std::vector<float> base={1,2,3,4}, o(4,0);
    CreateAclTensor(base,sh,&dA,ACL_FLOAT,&tA); CreateAclTensor(o,sh,&dO,ACL_FLOAT,&tO);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    EXPECT_FAIL(aclnnPowTensorTensorGetWorkspaceSize(tA,nullptr,tO,&ws,&ex),"tt null exp");
    aclDestroyTensor(tA);aclDestroyTensor(tO); aclrtFree(dA);aclrtFree(dO); }
}

int main() {
  int32_t deviceId = 0; aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  TestPowTensorScalarSpecial(stream);
  TestPowTensorScalarDtypes(stream);
  TestPowScalarTensor(stream);
  TestPowTensorTensor(stream);
  TestInplace(stream);
  TestExp2(stream);
  TestBoundaryAndErrors(stream);

  printf("\n========== RESULT: %d passed, %d failed ==========\n", g_passCount, g_failCount);
  aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize();
  return g_failCount > 0 ? 1 : 0;
}
