/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Comprehensive test for Cumsum operator (aclnnCumsum and aclnnCumsumV2)
 * Designed to achieve high line/branch coverage on op_api and op_host tiling.
 */

 #include <iostream>
 #include <vector>
 #include <cmath>
 #include <iomanip>
 #include <string>
 #include <algorithm>
 #include <numeric>
 #include <fstream>
 #include "acl/acl.h"
 #include "aclnnop/aclnn_cumsum.h"
 
 // ==================== 全局变量 ====================
 static int g_totalTests = 0;
 static int g_passedTests = 0;
 static std::ofstream g_csvFile;
 
 // ==================== 辅助函数 ====================
 #define CHECK_RET(cond, return_expr) \
     do { if (!(cond)) { return_expr; } } while (0)
 
 #define LOG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
 
 int64_t GetShapeSize(const std::vector<int64_t>& shape) {
     int64_t size = 1;
     for (auto d : shape) size *= d;
     return size;
 }
 
 int InitAcl(int32_t deviceId, aclrtStream* stream) {
     auto ret = aclInit(nullptr);
     CHECK_RET(ret == ACL_SUCCESS, return ret);
     ret = aclrtSetDevice(deviceId);
     CHECK_RET(ret == ACL_SUCCESS, return ret);
     ret = aclrtCreateStream(stream);
     CHECK_RET(ret == ACL_SUCCESS, return ret);
     return ACL_SUCCESS;
 }
 
 void FinalizeAcl(aclrtStream stream, int32_t deviceId) {
     aclrtDestroyStream(stream);
     aclrtResetDevice(deviceId);
     aclFinalize();
 }
 
 template<typename T>
 int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                     aclDataType dataType, void** deviceAddr, aclTensor** tensor) {
     int64_t size = GetShapeSize(shape) * sizeof(T);
     auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
     CHECK_RET(ret == ACL_SUCCESS, return ret);
     ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
     CHECK_RET(ret == ACL_SUCCESS, return ret);
 
     std::vector<int64_t> strides(shape.size(), 1);
     for (int64_t i = shape.size() - 2; i >= 0; --i) {
         strides[i] = shape[i + 1] * strides[i + 1];
     }
     *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                               strides.data(), 0, ACL_FORMAT_ND,
                               shape.data(), shape.size(), *deviceAddr);
     return ACL_SUCCESS;
 }
 
 const char* DtypeName(aclDataType dtype) {
     switch (dtype) {
         case ACL_FLOAT:   return "FLOAT32";
         case ACL_FLOAT16: return "FLOAT16";
         case ACL_BF16:    return "BF16";
         case ACL_INT32:   return "INT32";
         case ACL_INT64:   return "INT64";
         case ACL_INT8:    return "INT8";
         case ACL_UINT8:   return "UINT8";
         default:          return "UNKNOWN";
     }
 }
 
 // CPU 参考实现 (double累加)
 template<typename T>
 std::vector<double> CpuCumsum(const std::vector<T>& input, bool exclusive, bool reverse) {
     std::vector<double> result(input.size());
     if (reverse) {
         double sum = 0.0;
         for (int i = input.size() - 1; i >= 0; --i) {
             if (!exclusive) sum += static_cast<double>(input[i]);
             result[i] = sum;
             if (exclusive) sum += static_cast<double>(input[i]);
         }
     } else {
         double sum = 0.0;
         for (size_t i = 0; i < input.size(); ++i) {
             if (!exclusive) sum += static_cast<double>(input[i]);
             result[i] = sum;
             if (exclusive) sum += static_cast<double>(input[i]);
         }
     }
     return result;
 }
 
 // 结果比较 (浮点)
 template<typename T>
 bool CompareResults(const std::vector<T>& actual, const std::vector<double>& expected,
                     double atol, double rtol, double& maxError) {
     maxError = 0.0;
     for (size_t i = 0; i < actual.size(); ++i) {
         double act = static_cast<double>(actual[i]);
         double exp = expected[i];
         double absDiff = std::fabs(act - exp);
         double tol = atol + rtol * std::fabs(exp);
         if (absDiff > tol) {
             maxError = std::max(maxError, absDiff);
             return false;
         }
         maxError = std::max(maxError, absDiff);
     }
     return true;
 }
 
 // int8 溢出检测
 bool CompareInt8(const std::vector<int8_t>& actual, const std::vector<double>& expected,
                  double& maxError, bool& overflowDetected) {
     overflowDetected = false;
     maxError = 0.0;
     for (size_t i = 0; i < actual.size(); ++i) {
         int64_t exp = static_cast<int64_t>(expected[i]);
         int64_t act = actual[i];
         if (exp < -128 || exp > 127) overflowDetected = true;
         if (act != exp && (exp >= -128 && exp <= 127)) {
             maxError = std::max(maxError, static_cast<double>(std::abs(act - exp)));
             return false;
         }
     }
     return true;
 }
 
 // ==================== 核心测试执行器 ====================
 template<typename T>
 void RunCumsumTest(aclrtStream stream,
                    const std::string& testName,
                    const std::vector<T>& hostInput,
                    const std::vector<int64_t>& shape,
                    int64_t dim,
                    aclDataType dtype,
                    bool useV2,
                    bool exclusive,
                    bool reverse,
                    double atol, double rtol) {
     g_totalTests++;
     LOG_PRINT("\n[TEST] %s\n", testName.c_str());
 
     // 提前声明所有资源 (避免 goto 跨越初始化)
     int ret = 0;
     void* selfDevice = nullptr;
     void* outDevice = nullptr;
     aclTensor* selfTensor = nullptr;
     aclTensor* outTensor = nullptr;
     uint64_t workspaceSize = 0;
     aclOpExecutor* executor = nullptr;
     void* workspaceAddr = nullptr;
     std::vector<T> outHost(GetShapeSize(shape), T(0));
     double maxError = 0.0;
     bool passed = false;
     bool overflow = false;
 
     std::vector<double> expected = CpuCumsum(hostInput, exclusive, reverse);
 
     // 创建输入/输出张量
     ret = CreateAclTensor(hostInput, shape, dtype, &selfDevice, &selfTensor);
     if (ret != ACL_SUCCESS) { LOG_PRINT("  Create self failed\n"); goto cleanup; }
     ret = CreateAclTensor(outHost, shape, dtype, &outDevice, &outTensor);
     if (ret != ACL_SUCCESS) { LOG_PRINT("  Create out failed\n"); goto cleanup; }
 
     // 获取 workspace
     if (useV2) {
         ret = aclnnCumsumV2GetWorkspaceSize(selfTensor, dim, exclusive, reverse, outTensor, &workspaceSize, &executor);
     } else {
         ret = aclnnCumsumGetWorkspaceSize(selfTensor, dim, dtype, outTensor, &workspaceSize, &executor);
     }
     if (ret != ACL_SUCCESS) { LOG_PRINT("  GetWorkspaceSize failed: %d\n", ret); goto cleanup; }
 
     if (workspaceSize > 0) {
         ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
         if (ret != ACL_SUCCESS) { LOG_PRINT("  Malloc workspace failed\n"); goto cleanup; }
     }
 
     // 执行
     if (useV2) {
         ret = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, stream);
     } else {
         ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, stream);
     }
     if (ret != ACL_SUCCESS) { LOG_PRINT("  Execute failed: %d\n", ret); goto cleanup; }
 
     ret = aclrtSynchronizeStream(stream);
     if (ret != ACL_SUCCESS) { LOG_PRINT("  Stream sync failed\n"); goto cleanup; }
 
     ret = aclrtMemcpy(outHost.data(), outHost.size() * sizeof(T), outDevice,
                       outHost.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
     if (ret != ACL_SUCCESS) { LOG_PRINT("  Memcpy back failed\n"); goto cleanup; }
 
     // 验证结果
     if constexpr (std::is_same_v<T, int8_t>) {
         passed = CompareInt8(outHost, expected, maxError, overflow);
         if (overflow) LOG_PRINT("  [WARN] Integer overflow detected\n");
     } else {
         passed = CompareResults(outHost, expected, atol, rtol, maxError);
     }
 
     if (passed) {
         LOG_PRINT("  [PASS] Max error: %.6e\n", maxError);
         g_passedTests++;
     } else {
         LOG_PRINT("  [FAIL] Max error: %.6e (tol atol=%.2e, rtol=%.2e)\n", maxError, atol, rtol);
     }
 
     // 导出详细数据到 CSV（用于精度分析）
     if (g_csvFile.is_open()) {
         for (size_t i = 0; i < outHost.size(); ++i) {
             double act = static_cast<double>(outHost[i]);
             double exp = expected[i];
             double absErr = std::fabs(act - exp);
             double relErr = (std::fabs(exp) > 1e-12) ? absErr / std::fabs(exp) : absErr;
             g_csvFile << testName << "," << DtypeName(dtype) << "," << exclusive << "," << reverse << ","
                       << i << "," << act << "," << exp << "," << absErr << "," << relErr << "\n";
         }
     }
 
 cleanup:
     if (selfTensor) aclDestroyTensor(selfTensor);
     if (outTensor) aclDestroyTensor(outTensor);
     if (selfDevice) aclrtFree(selfDevice);
     if (outDevice) aclrtFree(outDevice);
     if (workspaceAddr) aclrtFree(workspaceAddr);
 }
 
 // ==================== 主函数 ====================
 int main() {
     // 初始化 CSV 文件
     g_csvFile.open("precision_data.csv");
     if (g_csvFile.is_open()) {
         g_csvFile << "test_name,dtype,exclusive,reverse,idx,actual,expected,abs_error,rel_error\n";
     }
 
     int32_t deviceId = 0;
     aclrtStream stream = nullptr;
     auto ret = InitAcl(deviceId, &stream);
     if (ret != ACL_SUCCESS) {
         LOG_PRINT("InitAcl failed\n");
         return -1;
     }
 
     // ==================== 1. API 层基础测试 (aclnnCumsum) ====================
     // A1: 空指针已在异常测试中单独验证，此处正常流程
     // A2~A4: 数据类型支持测试（通过不同 dtype 触发）
     RunCumsumTest<float>(stream, "Base_1D_float32", std::vector<float>(100, 1.0f), {100}, 0,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     RunCumsumTest<uint16_t>(stream, "Base_1D_float16", std::vector<uint16_t>(100, 0x3c00), {100}, 0,
                             ACL_FLOAT16, false, false, false, 1e-3, 1e-3);
     RunCumsumTest<int32_t>(stream, "Base_1D_int32", std::vector<int32_t>(100, 1), {100}, 0,
                            ACL_INT32, false, false, false, 0.0, 0.0);
     RunCumsumTest<int64_t>(stream, "Base_1D_int64", std::vector<int64_t>(100, 1), {100}, 0,
                            ACL_INT64, false, false, false, 0.0, 0.0);
     // BF16 需芯片支持，若有则取消注释
     // RunCumsumTest<uint16_t>(stream, "Base_1D_bf16", std::vector<uint16_t>(100, 0x3f80), {100}, 0,
     //                         ACL_BF16, false, false, false, 1e-3, 1e-3);
 
     // A5: self 与 out 类型不一致 → 应在 GetWorkspaceSize 阶段失败，单独测试
     {
         LOG_PRINT("\n[TEST] Negative: dtype mismatch\n");
         std::vector<float> inData = {1,2,3};
         std::vector<int32_t> outData = {0,0,0};
         void* inDev = nullptr, *outDev = nullptr;
         aclTensor* inTensor = nullptr, *outTensor = nullptr;
         CreateAclTensor(inData, {3}, ACL_FLOAT, &inDev, &inTensor);
         CreateAclTensor(outData, {3}, ACL_INT32, &outDev, &outTensor);
         uint64_t ws = 0;
         aclOpExecutor* exec = nullptr;
         ret = aclnnCumsumGetWorkspaceSize(inTensor, 0, ACL_FLOAT, outTensor, &ws, &exec);
         LOG_PRINT("  Expected error (dtype mismatch), got %d %s\n", ret, (ret!=0)?"[PASS]":"[FAIL]");
         if (ret != ACL_SUCCESS) g_passedTests++;
         g_totalTests++;
         aclDestroyTensor(inTensor); aclDestroyTensor(outTensor);
         aclrtFree(inDev); aclrtFree(outDev);
     }
 
     // A6: 0维标量
     RunCumsumTest<float>(stream, "Scalar_0D", {42.0f}, {}, 0, ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // A7~A8: dim 边界
     RunCumsumTest<float>(stream, "Dim_out_of_range_high", std::vector<float>(10,1.0f), {10}, 5,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);  // 应失败
     RunCumsumTest<float>(stream, "Dim_out_of_range_low", std::vector<float>(10,1.0f), {10}, -6,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // A9: shape 不一致
     {
         LOG_PRINT("\n[TEST] Negative: shape mismatch\n");
         std::vector<float> inData = {1,2,3};
         void* inDev=nullptr, *outDev=nullptr;
         aclTensor* inTensor=nullptr, *outTensor=nullptr;
         CreateAclTensor(inData, {3}, ACL_FLOAT, &inDev, &inTensor);
         CreateAclTensor(inData, {2}, ACL_FLOAT, &outDev, &outTensor);
         uint64_t ws=0; aclOpExecutor* exec=nullptr;
         ret = aclnnCumsumGetWorkspaceSize(inTensor, 0, ACL_FLOAT, outTensor, &ws, &exec);
         LOG_PRINT("  Expected error (shape mismatch), got %d %s\n", ret, (ret!=0)?"[PASS]":"[FAIL]");
         if (ret != ACL_SUCCESS) g_passedTests++;
         g_totalTests++;
         aclDestroyTensor(inTensor); aclDestroyTensor(outTensor);
         aclrtFree(inDev); aclrtFree(outDev);
     }
     // A10: 维度超过8
     {
         std::vector<int64_t> bigShape(9, 2);
         std::vector<float> bigData(512, 1.0f);
         void* dev=nullptr; aclTensor* bigTensor=nullptr;
         CreateAclTensor(bigData, bigShape, ACL_FLOAT, &dev, &bigTensor);
         uint64_t ws=0; aclOpExecutor* exec=nullptr;
         ret = aclnnCumsumGetWorkspaceSize(bigTensor, 0, ACL_FLOAT, bigTensor, &ws, &exec);
         LOG_PRINT("  Dim>8 should fail: %d %s\n", ret, (ret!=0)?"[PASS]":"[FAIL]");
         if (ret != ACL_SUCCESS) g_passedTests++;
         g_totalTests++;
         aclDestroyTensor(bigTensor); aclrtFree(dev);
     }
     // A11: 空张量
     RunCumsumTest<float>(stream, "Empty_tensor", std::vector<float>(), {0}, 0,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // A12: 普通路径（不满足 Cube）
     RunCumsumTest<float>(stream, "Normal_path", std::vector<float>(100,1.0f), {10,10}, 1,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // A13: Cube 路径 (需要最后 dim ≥512 且 batch≥12800)
     {
         std::vector<float> bigData(12800*512, 1.0f);
         RunCumsumTest<float>(stream, "Cube_path", bigData, {12800,512}, 1,
                              ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     }
     // A15: V2 API
     std::vector<float> v2data = {1,2,3,4,5};
     RunCumsumTest<float>(stream, "V2_normal", v2data, {5}, 0,
                          ACL_FLOAT, true, false, false, 1e-5, 1e-5);
     RunCumsumTest<float>(stream, "V2_exclusive_true", v2data, {5}, 0,
                          ACL_FLOAT, true, true, false, 1e-5, 1e-5);
     RunCumsumTest<float>(stream, "V2_reverse_true", v2data, {5}, 0,
                          ACL_FLOAT, true, false, true, 1e-5, 1e-5);
     RunCumsumTest<float>(stream, "V2_both_true", v2data, {5}, 0,
                          ACL_FLOAT, true, true, true, 1e-5, 1e-5);
 
     // ==================== 2. cumsum.cpp 分支 (AiCore/AiCpu) ====================
     // B1~B4: 不同芯片+数据类型组合由实际硬件决定，这里通过 shape 确保正常路径
     RunCumsumTest<float>(stream, "AiCore_float", std::vector<float>(100,1.0f), {100}, 0,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     RunCumsumTest<int32_t>(stream, "AiCore_int32", std::vector<int32_t>(100,1), {100}, 0,
                            ACL_INT32, false, false, false, 0.0, 0.0);
     // 其他类型如 int64 走 AiCpu，这里也覆盖
     RunCumsumTest<int64_t>(stream, "AiCpu_int64", std::vector<int64_t>(100,1), {100}, 0,
                            ACL_INT64, false, false, false, 0.0, 0.0);
 
     // ==================== 3. Tiling 层分发 (cumsum_tiling.cpp) ====================
     // C1: 浮点 → TilingCumsumAscendc，C2: 整数 → TilingCumsum4Int
     RunCumsumTest<float>(stream, "Tiling_float", std::vector<float>(256,1.0f), {16,16}, 1,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     RunCumsumTest<int32_t>(stream, "Tiling_int", std::vector<int32_t>(256,1), {16,16}, 1,
                            ACL_INT32, false, false, false, 0.0, 0.0);
 
     // ==================== 4. 浮点 Tiling 详细分支 (cumsum_tiling_ascendc_arch35.cpp) ====================
     // 以下 shape 均针对 ascend910_93 典型值设计 (clSize=64, ubSize~256KB, coreNum=64)
     // D1: N≥cl, R全载, M≥core
     RunCumsumTest<float>(stream, "FloatTiling_NGtCl_RFull_MGeCore",
                          std::vector<float>(128*64*64, 1.0f), {128,64,64}, 2,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // D2: N≥cl, R全载, M<core 借N
     RunCumsumTest<float>(stream, "FloatTiling_NGtCl_RFull_MLtCore_BorrowN",
                          std::vector<float>(32*64*64, 1.0f), {32,64,64}, 2,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // D3: N≥cl, R不能全载, M够分核
     RunCumsumTest<float>(stream, "FloatTiling_NGtCl_RNotFull_MGeCore",
                          std::vector<float>(128*1024*64, 1.0f), {128,1024,64}, 2,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // D5: N≥cl, R不能全载, 借R
     RunCumsumTest<float>(stream, "FloatTiling_NGtCl_RNotFull_BorrowR",
                          std::vector<float>(8*1024*64, 1.0f), {8,1024,64}, 2,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // D6: N<cl, R*N≥cl, 单向, R全载
     RunCumsumTest<float>(stream, "FloatTiling_NLtCl_RNGeCl_OneWay_RFull",
                          std::vector<float>(64*16*8, 1.0f), {64,16,8}, 1,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // D7: 单向, R不能全载, M够切
     RunCumsumTest<float>(stream, "FloatTiling_OneWay_RNotFull_MGeCore",
                          std::vector<float>(128*256*8, 1.0f), {128,256,8}, 1,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // D8: 单向, R不能全载, M不够借R
     RunCumsumTest<float>(stream, "FloatTiling_OneWay_RNotFull_MLtCore_BorrowR",
                          std::vector<float>(8*256*8, 1.0f), {8,256,8}, 1,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // D9: 双向 sklansky, R全载 (需要 alignN 较小，通过小的 N 实现)
     RunCumsumTest<float>(stream, "FloatTiling_TwoWay_RFull",
                          std::vector<float>(64*512*4, 1.0f), {64,512,4}, 1,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // D10: 双向, R不能全载, M够切
     RunCumsumTest<float>(stream, "FloatTiling_TwoWay_RNotFull_MGeCore",
                          std::vector<float>(128*4096*4, 1.0f), {128,4096,4}, 1,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // D11: 双向, R不能全载, M不够借R
     RunCumsumTest<float>(stream, "FloatTiling_TwoWay_RNotFull_MLtCore_BorrowR",
                          std::vector<float>(8*4096*4, 1.0f), {8,4096,4}, 1,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // D12: M*R*N ≥ cl, 进入 MRNGreaterCl
     RunCumsumTest<float>(stream, "FloatTiling_MRNGeCl",
                          std::vector<float>(64*32*16, 1.0f), {64,32,16}, 1,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // D13: M*R*N < cl, 简单路径
     RunCumsumTest<float>(stream, "FloatTiling_MRNLtCl",
                          std::vector<float>(4*4*4, 1.0f), {4,4,4}, 1,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     // D14: 空 tensor（已在 A11 覆盖）
 
     // ==================== 5. 整数 Tiling 分支 (cumsum_tiling_ascendc_int_arch35.cpp) ====================
     // E1: axis 第一维
     RunCumsumTest<int32_t>(stream, "IntTiling_axis0", std::vector<int32_t>(5*3*2, 1), {5,3,2}, 0,
                            ACL_INT32, false, false, false, 0.0, 0.0);
     // E2: axis 最后一维
     RunCumsumTest<int32_t>(stream, "IntTiling_axisLast", std::vector<int32_t>(2*3*5, 1), {2,3,5}, 2,
                            ACL_INT32, false, false, false, 0.0, 0.0);
     // E3: axis 中间
     RunCumsumTest<int32_t>(stream, "IntTiling_axisMid", std::vector<int32_t>(2*4*3, 1), {2,4,3}, 1,
                            ACL_INT32, false, false, false, 0.0, 0.0);
     // E4: 负数轴
     RunCumsumTest<int32_t>(stream, "IntTiling_axisNeg", std::vector<int32_t>(2*4*3, 1), {2,4,3}, -2,
                            ACL_INT32, false, false, false, 0.0, 0.0);
     // E6: 右轴大 (rightAxisLen*dtypeSize >= vlSize_/2)
     RunCumsumTest<int32_t>(stream, "IntTiling_rightAxisLarge", std::vector<int32_t>(2*4*1024, 1), {2,4,1024}, 1,
                            ACL_INT32, false, false, false, 0.0, 0.0);
     // E7: 左轴大, 触发 TD leftA
     RunCumsumTest<int32_t>(stream, "IntTiling_leftAxisLarge", std::vector<int32_t>(128*4*8, 1), {128,4,8}, 1,
                            ACL_INT32, false, false, false, 0.0, 0.0);
     // E8: 左轴不够大, 进入 TD R
     RunCumsumTest<int32_t>(stream, "IntTiling_leftAxisSmall", std::vector<int32_t>(4*128*8, 1), {4,128,8}, 1,
                            ACL_INT32, false, false, false, 0.0, 0.0);
     // E12~E14: 分核轴选择由 GetMCTilingInfo 内部权重决定，通过不同 shape 间接覆盖
     RunCumsumTest<int32_t>(stream, "IntTiling_splitLA", std::vector<int32_t>(64*64*8, 1), {64,64,8}, 1,
                            ACL_INT32, false, false, false, 0.0, 0.0);
     RunCumsumTest<int32_t>(stream, "IntTiling_splitRA", std::vector<int32_t>(4*64*64, 1), {4,64,64}, 1,
                            ACL_INT32, false, false, false, 0.0, 0.0);
     RunCumsumTest<int32_t>(stream, "IntTiling_splitR", std::vector<int32_t>(4*128*4, 1), {4,128,4}, 1,
                            ACL_INT32, false, false, false, 0.0, 0.0);
     // E15: 三种 tilingKey
     // CUM_WITH_GROUP (isRBlockAxis=1) 已在 splitR 中覆盖
     // CUM_NO_SPLIT (rightAxisLen*dtypeSize >= vlSize_)
     RunCumsumTest<int32_t>(stream, "IntTiling_noSplit", std::vector<int32_t>(4*1024*1, 1), {4,1024,1}, 1,
                            ACL_INT32, false, false, false, 0.0, 0.0);
     // CUM_AR_SPLIT 其他情况
 
     // E17: Bank group conflict 通过特定 shape 触发 (已在 AdjustTensor4TDLA 中)
     // E18: CalcAxisWeight 已通过上述用例覆盖
 
     // ==================== 精度分析专用用例（输出详细CSV） ====================
     std::vector<float> longOnes(10000, 1.0f);
     RunCumsumTest<float>(stream, "Precision_LongOnes", longOnes, {10000}, 0,
                          ACL_FLOAT, false, false, false, 1e-4, 1e-5);
     std::vector<float> mixedMag;
     for (int i = 0; i < 1000; ++i) { mixedMag.push_back(1e8f); mixedMag.push_back(1e-6f); }
     RunCumsumTest<float>(stream, "Precision_MixedMag", mixedMag, {2000}, 0,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     std::vector<float> altSign;
     for (int i = 0; i < 10001; ++i) altSign.push_back((i%2==0)?1.0f:-1.0f);
     RunCumsumTest<float>(stream, "Precision_AltSign", altSign, {10001}, 0,
                          ACL_FLOAT, false, false, false, 1e-5, 1e-5);
     std::vector<float> many01(10000, 0.1f);
     RunCumsumTest<float>(stream, "Precision_Many01", many01, {10000}, 0,
                          ACL_FLOAT, false, false, false, 1e-3, 1e-5);
     std::vector<int8_t> overflowInt8 = {100, 50, 50};
     RunCumsumTest<int8_t>(stream, "Precision_Int8Overflow", overflowInt8, {3}, 0,
                           ACL_INT8, false, false, false, 0.0, 0.0);
 
     // ==================== 异常输入（空指针等） ====================
     {
         LOG_PRINT("\n[TEST] Negative: nullptr self\n");
         uint64_t ws=0; aclOpExecutor* exec=nullptr;
         ret = aclnnCumsumGetWorkspaceSize(nullptr, 0, ACL_FLOAT, nullptr, &ws, &exec);
         LOG_PRINT("  Return %d %s\n", ret, (ret!=0)?"[PASS]":"[FAIL]");
         if (ret != ACL_SUCCESS) g_passedTests++;
         g_totalTests++;
     }
     {
         LOG_PRINT("\n[TEST] Negative: unsupported dtype (BOOL)\n");
         std::vector<int8_t> boolData = {1,0,1};
         void* dev=nullptr; aclTensor* t=nullptr;
         CreateAclTensor(boolData, {3}, ACL_BOOL, &dev, &t);
         uint64_t ws=0; aclOpExecutor* exec=nullptr;
         ret = aclnnCumsumGetWorkspaceSize(t, 0, ACL_BOOL, t, &ws, &exec);
         LOG_PRINT("  Return %d %s\n", ret, (ret!=0)?"[PASS]":"[FAIL]");
         if (ret != ACL_SUCCESS) g_passedTests++;
         g_totalTests++;
         aclDestroyTensor(t); aclrtFree(dev);
     }
 
     // 清理与汇总
     FinalizeAcl(stream, deviceId);
     LOG_PRINT("\n========== Summary ==========\n");
     LOG_PRINT("Total tests: %d, Passed: %d, Failed: %d\n",
               g_totalTests, g_passedTests, g_totalTests - g_passedTests);
     if (g_csvFile.is_open()) g_csvFile.close();
     return (g_passedTests == g_totalTests) ? 0 : 1;
 }