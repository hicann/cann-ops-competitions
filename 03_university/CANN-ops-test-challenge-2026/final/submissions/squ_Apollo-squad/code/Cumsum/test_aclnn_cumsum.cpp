// /**
//  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
//  * CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
//  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// #include <algorithm>
// #include <cmath>
// #include <cstdint>
// #include <cstring>
// #include <iostream>
// #include <numeric>
// #include <string>
// #include <vector>
// #include "acl/acl.h"
// #include "aclnnop/aclnn_cumsum.h"

// #define CHECK_RET(cond, return_expr) \
//   do {                               \
//     if (!(cond)) {                   \
//       return_expr;                   \
//     }                                \
//   } while (0)

// #define LOG_PRINT(message, ...)     \
//   do {                              \
//     printf(message, ##__VA_ARGS__); \
//   } while (0)

// static int gPass = 0;
// static int gFail = 0;
// static aclrtStream gStream = nullptr;
// static constexpr int32_t kDeviceId = 0;

// static void Pass(const char* name) {
//   LOG_PRINT("[PASS] %s\n", name);
//   gPass++;
// }

// static void Fail(const char* name, const char* why = "") {
//   LOG_PRINT("[FAIL] %s (%s)\n", name, why);
//   gFail++;
// }

// static int64_t ShapeProd(const std::vector<int64_t>& shape) {
//   int64_t n = 1;
//   for (auto d : shape) n *= d;
//   return n;
// }

// static int Init() {
//   auto ret = aclInit(nullptr);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. %d\n", ret); return ret);
//   ret = aclrtSetDevice(kDeviceId);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. %d\n", ret); return ret);
//   ret = aclrtCreateStream(&gStream);
//   CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. %d\n", ret); return ret);
//   return 0;
// }

// static void Finalize() {
//   if (gStream) {
//     aclrtDestroyStream(gStream);
//   }
//   aclrtResetDevice(kDeviceId);
//   aclFinalize();
// }

// static size_t DTypeSize(aclDataType dt) {
//   switch (dt) {
//     case ACL_FLOAT:
//       return 4;
//     case ACL_FLOAT16:
//       return 2;
//     case ACL_BF16:
//       return 2;
//     case ACL_INT32:
//       return 4;
//     case ACL_INT64:
//       return 8;
//     case ACL_INT16:
//       return 2;
//     case ACL_INT8:
//       return 1;
//     case ACL_UINT8:
//       return 1;
//     case ACL_BOOL:
//       return 1;
//     case ACL_DOUBLE:
//       return 8;
//     default:
//       return 4;
//   }
// }

// static uint16_t F32ToFP16(float f) {
//   union {
//     float f;
//     uint32_t u;
//   } v{f};
//   uint32_t b = v.u;
//   uint16_t sign = static_cast<uint16_t>((b >> 16u) & 0x8000u);
//   int exp = static_cast<int>((b >> 23u) & 0xffu) - 127 + 15;
//   uint32_t mant = b & 0x7fffffu;
//   if (exp <= 0) return sign;
//   if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
//   return static_cast<uint16_t>(sign | static_cast<uint16_t>(exp << 10) | static_cast<uint16_t>(mant >> 13));
// }

// static float FP16ToF32(uint16_t h) {
//   uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
//   uint32_t exp = (h >> 10) & 0x1fu;
//   uint32_t mant = h & 0x3ffu;
//   if (!exp && !mant) return 0.f;
//   if (exp == 31) {
//     union {
//       uint32_t u;
//       float f;
//     } v{sign | 0x7f800000u | (mant << 13)};
//     return v.f;
//   }
//   if (!exp) {
//     while (!(mant & 0x400u)) {
//       mant <<= 1;
//       exp--;
//     }
//     exp++;
//     mant &= 0x3ffu;
//   }
//   union {
//     uint32_t u;
//     float f;
//   } v{sign | ((exp + 112u) << 23) | (mant << 13)};
//   return v.f;
// }

// static uint16_t F32ToBF16(float f) {
//   union {
//     float f;
//     uint32_t u;
//   } v{f};
//   return static_cast<uint16_t>(v.u >> 16);
// }

// static float BF16ToF32(uint16_t b) {
//   union {
//     uint32_t u;
//     float f;
//   } v{static_cast<uint32_t>(b) << 16};
//   return v.f;
// }

// static std::vector<uint8_t> PackData(const std::vector<double>& src, aclDataType dt) {
//   size_t n = src.size();
//   size_t bytes = DTypeSize(dt);
//   std::vector<uint8_t> out(n * bytes);
//   for (size_t i = 0; i < n; ++i) {
//     void* p = out.data() + i * bytes;
//     switch (dt) {
//       case ACL_FLOAT: {
//         float v = static_cast<float>(src[i]);
//         std::memcpy(p, &v, 4);
//         break;
//       }
//       case ACL_FLOAT16: {
//         uint16_t v = F32ToFP16(static_cast<float>(src[i]));
//         std::memcpy(p, &v, 2);
//         break;
//       }
//       case ACL_BF16: {
//         uint16_t v = F32ToBF16(static_cast<float>(src[i]));
//         std::memcpy(p, &v, 2);
//         break;
//       }
//       case ACL_INT32: {
//         int32_t v = static_cast<int32_t>(src[i]);
//         std::memcpy(p, &v, 4);
//         break;
//       }
//       case ACL_INT64: {
//         int64_t v = static_cast<int64_t>(src[i]);
//         std::memcpy(p, &v, 8);
//         break;
//       }
//       case ACL_INT16: {
//         int16_t v = static_cast<int16_t>(src[i]);
//         std::memcpy(p, &v, 2);
//         break;
//       }
//       case ACL_INT8: {
//         int8_t v = static_cast<int8_t>(src[i]);
//         std::memcpy(p, &v, 1);
//         break;
//       }
//       case ACL_UINT8: {
//         uint8_t v = static_cast<uint8_t>(src[i]);
//         std::memcpy(p, &v, 1);
//         break;
//       }
//       case ACL_BOOL: {
//         uint8_t v = (src[i] != 0.0) ? 1 : 0;
//         std::memcpy(p, &v, 1);
//         break;
//       }
//       case ACL_DOUBLE: {
//         double v = src[i];
//         std::memcpy(p, &v, 8);
//         break;
//       }
//       default:
//         break;
//     }
//   }
//   return out;
// }

// static std::vector<double> UnpackData(void* devAddr, int64_t n, aclDataType dt) {
//   size_t bytes = DTypeSize(dt);
//   std::vector<uint8_t> buf(static_cast<size_t>(n) * bytes);
//   aclrtMemcpy(buf.data(), buf.size(), devAddr, buf.size(), ACL_MEMCPY_DEVICE_TO_HOST);
//   std::vector<double> out(static_cast<size_t>(n), 0.0);
//   for (int64_t i = 0; i < n; ++i) {
//     const void* p = buf.data() + static_cast<size_t>(i) * bytes;
//     switch (dt) {
//       case ACL_FLOAT: {
//         float v;
//         std::memcpy(&v, p, 4);
//         out[static_cast<size_t>(i)] = v;
//         break;
//       }
//       case ACL_FLOAT16: {
//         uint16_t v;
//         std::memcpy(&v, p, 2);
//         out[static_cast<size_t>(i)] = FP16ToF32(v);
//         break;
//       }
//       case ACL_BF16: {
//         uint16_t v;
//         std::memcpy(&v, p, 2);
//         out[static_cast<size_t>(i)] = BF16ToF32(v);
//         break;
//       }
//       case ACL_INT32: {
//         int32_t v;
//         std::memcpy(&v, p, 4);
//         out[static_cast<size_t>(i)] = static_cast<double>(v);
//         break;
//       }
//       case ACL_INT64: {
//         int64_t v;
//         std::memcpy(&v, p, 8);
//         out[static_cast<size_t>(i)] = static_cast<double>(v);
//         break;
//       }
//       case ACL_INT16: {
//         int16_t v;
//         std::memcpy(&v, p, 2);
//         out[static_cast<size_t>(i)] = static_cast<double>(v);
//         break;
//       }
//       case ACL_INT8: {
//         int8_t v;
//         std::memcpy(&v, p, 1);
//         out[static_cast<size_t>(i)] = static_cast<double>(v);
//         break;
//       }
//       case ACL_UINT8: {
//         uint8_t v;
//         std::memcpy(&v, p, 1);
//         out[static_cast<size_t>(i)] = static_cast<double>(v);
//         break;
//       }
//       case ACL_BOOL: {
//         uint8_t v;
//         std::memcpy(&v, p, 1);
//         out[static_cast<size_t>(i)] = static_cast<double>(v);
//         break;
//       }
//       case ACL_DOUBLE: {
//         double v;
//         std::memcpy(&v, p, 8);
//         out[static_cast<size_t>(i)] = v;
//         break;
//       }
//       default:
//         break;
//     }
//   }
//   return out;
// }

// static int CreateTensor(const std::vector<double>& hostData, const std::vector<int64_t>& shape, aclDataType dt,
//                         void** devAddr, aclTensor** tensor) {
//   int64_t n = ShapeProd(shape);
//   size_t bytes = static_cast<size_t>(n) * DTypeSize(dt);
//   if (bytes == 0) {
//     *devAddr = nullptr;
//   } else {
//     auto packed = PackData(hostData, dt);
//     auto ret = aclrtMalloc(devAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. %d\n", ret); return ret);
//     ret = aclrtMemcpy(*devAddr, bytes, packed.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
//     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. %d\n", ret); return ret);
//   }

//   if (shape.empty()) {
//     *tensor = aclCreateTensor(nullptr, 0, dt, nullptr, 0, aclFormat::ACL_FORMAT_ND, nullptr, 0, *devAddr);
//   } else {
//     std::vector<int64_t> strides(shape.size(), 1);
//     for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
//       strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
//     }
//     *tensor = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
//                               shape.size(), *devAddr);
//   }
//   CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);
//   return 0;
// }

// static void DestroyTensor(aclTensor* t, void* addr) {
//   if (t) aclDestroyTensor(t);
//   if (addr) aclrtFree(addr);
// }

// static void CpuCumsum(const std::vector<double>& in, std::vector<double>& out, const std::vector<int64_t>& shape,
//                       int64_t dim, bool exclusive = false, bool reverse = false) {
//   int nd = static_cast<int>(shape.size());
//   int64_t n = ShapeProd(shape);
//   out.assign(static_cast<size_t>(n), 0.0);
//   if (n == 0) return;
//   if (nd == 0) {
//     out = in;
//     return;
//   }
//   if (dim < 0) dim += nd;

//   int64_t outer = 1;
//   int64_t dimSz = shape[static_cast<size_t>(dim)];
//   int64_t inner = 1;
//   for (int i = 0; i < dim; ++i) outer *= shape[static_cast<size_t>(i)];
//   for (int i = static_cast<int>(dim) + 1; i < nd; ++i) inner *= shape[static_cast<size_t>(i)];

//   for (int64_t o = 0; o < outer; ++o) {
//     for (int64_t x = 0; x < inner; ++x) {
//       int64_t base = o * dimSz * inner + x;
//       double sum = 0.0;
//       if (!reverse) {
//         for (int64_t k = 0; k < dimSz; ++k) {
//           int64_t idx = base + k * inner;
//           if (exclusive) {
//             out[static_cast<size_t>(idx)] = sum;
//             sum += in[static_cast<size_t>(idx)];
//           } else {
//             sum += in[static_cast<size_t>(idx)];
//             out[static_cast<size_t>(idx)] = sum;
//           }
//         }
//       } else {
//         for (int64_t k = dimSz - 1; k >= 0; --k) {
//           int64_t idx = base + k * inner;
//           if (exclusive) {
//             out[static_cast<size_t>(idx)] = sum;
//             sum += in[static_cast<size_t>(idx)];
//           } else {
//             sum += in[static_cast<size_t>(idx)];
//             out[static_cast<size_t>(idx)] = sum;
//           }
//         }
//       }
//     }
//   }
// }

// static bool CompareFloat(const std::vector<double>& actual, const std::vector<double>& expected, double atol, double rtol,
//                          bool printStat = false) {
//   double maxErr = 0.0;
//   int64_t maxPos = -1;
//   bool ok = true;
//   for (size_t i = 0; i < actual.size(); ++i) {
//     double err = std::abs(actual[i] - expected[i]);
//     double tol = atol + rtol * std::abs(expected[i]);
//     if (err > maxErr) {
//       maxErr = err;
//       maxPos = static_cast<int64_t>(i);
//     }
//     if (err > tol) ok = false;
//   }
//   if (printStat || !ok) {
//     LOG_PRINT("  Max error: %.8e at %lld\n", maxErr, static_cast<long long>(maxPos));
//   }
//   return ok;
// }

// static bool CompareExact(const std::vector<double>& actual, const std::vector<double>& expected) {
//   for (size_t i = 0; i < actual.size(); ++i) {
//     if (static_cast<int64_t>(actual[i]) != static_cast<int64_t>(expected[i])) return false;
//   }
//   return true;
// }

// struct TestCase {
//   const char* name;
//   std::vector<int64_t> shape;
//   int64_t dim = 0;
//   std::vector<double> input;
//   aclDataType selfDt = ACL_FLOAT;
//   aclDataType outDt = ACL_FLOAT;
//   bool exclusive = false;
//   bool reverse = false;
//   bool useV2 = false;
//   double atol = 1e-5;
//   double rtol = 1e-5;
//   bool isInt = false;
//   bool printStat = false;
// };

// static bool RunCase(const TestCase& tc) {
//   int64_t n = ShapeProd(tc.shape);
//   void *selfAddr = nullptr, *outAddr = nullptr;
//   aclTensor *self = nullptr, *out = nullptr;
//   if (CreateTensor(tc.input, tc.shape, tc.selfDt, &selfAddr, &self) != 0) {
//     Fail(tc.name, "create self failed");
//     return false;
//   }
//   std::vector<double> outInit(static_cast<size_t>(n), 0.0);
//   if (CreateTensor(outInit, tc.shape, tc.outDt, &outAddr, &out) != 0) {
//     DestroyTensor(self, selfAddr);
//     Fail(tc.name, "create out failed");
//     return false;
//   }

//   uint64_t workspaceSize = 0;
//   aclOpExecutor* executor = nullptr;
//   aclnnStatus ret = ACL_SUCCESS;
//   if (!tc.useV2) {
//     ret = aclnnCumsumGetWorkspaceSize(self, tc.dim, tc.outDt, out, &workspaceSize, &executor);
//   } else {
//     ret = aclnnCumsumV2GetWorkspaceSize(self, tc.dim, tc.exclusive, tc.reverse, out, &workspaceSize, &executor);
//   }
//   if (ret != ACL_SUCCESS) {
//     DestroyTensor(self, selfAddr);
//     DestroyTensor(out, outAddr);
//     Fail(tc.name, "get workspace failed");
//     return false;
//   }

//   void* workspaceAddr = nullptr;
//   if (workspaceSize > 0) {
//     ret = static_cast<aclnnStatus>(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
//     if (ret != ACL_SUCCESS) {
//       DestroyTensor(self, selfAddr);
//       DestroyTensor(out, outAddr);
//       Fail(tc.name, "malloc workspace failed");
//       return false;
//     }
//   }

//   if (!tc.useV2) {
//     ret = aclnnCumsum(workspaceAddr, workspaceSize, executor, gStream);
//   } else {
//     ret = aclnnCumsumV2(workspaceAddr, workspaceSize, executor, gStream);
//   }
//   aclrtSynchronizeStream(gStream);

//   bool ok = false;
//   if (ret != ACL_SUCCESS) {
//     Fail(tc.name, "exec failed");
//   } else if (n == 0) {
//     Pass(tc.name);
//     ok = true;
//   } else {
//     auto actual = UnpackData(outAddr, n, tc.outDt);
//     std::vector<double> expected;
//     CpuCumsum(tc.input, expected, tc.shape, tc.dim, tc.exclusive, tc.reverse);
//     ok = tc.isInt ? CompareExact(actual, expected) : CompareFloat(actual, expected, tc.atol, tc.rtol, tc.printStat);
//     if (ok) {
//       Pass(tc.name);
//     } else {
//       Fail(tc.name, "result mismatch");
//     }
//   }

//   DestroyTensor(self, selfAddr);
//   DestroyTensor(out, outAddr);
//   if (workspaceAddr) aclrtFree(workspaceAddr);
//   return ok;
// }

// static bool RunErrV1(const char* name, aclTensor* self, int64_t dim, aclDataType dtype, aclTensor* out) {
//   uint64_t ws = 0;
//   aclOpExecutor* exe = nullptr;
//   auto ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &ws, &exe);
//   if (ret != ACL_SUCCESS) {
//     Pass(name);
//     return true;
//   }
//   Fail(name, "should fail but success");
//   return false;
// }

// static bool RunErrV2(const char* name, aclTensor* self, int64_t dim, bool exclusive, bool reverse, aclTensor* out) {
//   uint64_t ws = 0;
//   aclOpExecutor* exe = nullptr;
//   auto ret = aclnnCumsumV2GetWorkspaceSize(self, dim, exclusive, reverse, out, &ws, &exe);
//   if (ret != ACL_SUCCESS) {
//     Pass(name);
//     return true;
//   }
//   Fail(name, "should fail but success");
//   return false;
// }

// static std::vector<double> Seq(int64_t n, double start = 1.0) {
//   std::vector<double> v(static_cast<size_t>(n));
//   std::iota(v.begin(), v.end(), start);
//   return v;
// }

// static std::vector<double> Fill(int64_t n, double x) { return std::vector<double>(static_cast<size_t>(n), x); }

// int main() {
//   if (Init() != 0) return -1;
//   LOG_PRINT("========= Cumsum high-coverage tests =========\n");

//   RunCase({"TC01_f32_1d", {8}, 0, Seq(8), ACL_FLOAT, ACL_FLOAT});
//   RunCase({"TC02_f32_2d_d1", {2, 8}, 1, Seq(16), ACL_FLOAT, ACL_FLOAT});
//   RunCase({"TC03_fp16", {64}, 0, Seq(64), ACL_FLOAT, ACL_FLOAT16, false, false, false, 2e-2, 2e-2});
//   RunCase({"TC04_bf16", {64}, 0, Seq(64), ACL_FLOAT, ACL_BF16, false, false, false, 5e-2, 5e-2});
//   RunCase({"TC05_i32", {64}, 0, Seq(64), ACL_INT32, ACL_INT32, false, false, false, 0.0, 0.0, true});
//   RunCase({"TC06_i64", {64}, 0, Seq(64), ACL_INT64, ACL_INT64, false, false, false, 0.0, 0.0, true});
//   RunCase({"TC07_i8", {64}, 0, Fill(64, 1.0), ACL_INT8, ACL_INT8, false, false, false, 0.0, 0.0, true});
//   RunCase({"TC08_u8", {64}, 0, Fill(64, 1.0), ACL_UINT8, ACL_UINT8, false, false, false, 0.0, 0.0, true});

//   RunCase({"TC09_dim_neg1", {2, 8}, -1, Seq(16), ACL_FLOAT, ACL_FLOAT});
//   RunCase({"TC10_dim_neg2", {2, 8}, -2, Seq(16), ACL_FLOAT, ACL_FLOAT});
//   RunCase({"TC11_3d_d0", {2, 3, 4}, 0, Seq(24), ACL_FLOAT, ACL_FLOAT});
//   RunCase({"TC12_3d_d1", {2, 3, 4}, 1, Seq(24), ACL_FLOAT, ACL_FLOAT});
//   RunCase({"TC13_3d_d2", {2, 3, 4}, 2, Seq(24), ACL_FLOAT, ACL_FLOAT});

//   RunCase({"TC14_zero", {128}, 0, Fill(128, 0.0), ACL_FLOAT, ACL_FLOAT});
//   RunCase({"TC15_negative", {128}, 0, Fill(128, -1.0), ACL_FLOAT, ACL_FLOAT});
//   RunCase({"TC16_alt_pm", {128}, 0, {1, -1, 1, -1, 1, -1, 1, -1}, ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-5,
//            1e-5});

//   RunCase({"TC17_empty", {0, 32}, 0, {}, ACL_FLOAT, ACL_FLOAT});
//   RunCase({"TC18_scalar_f32", {}, 0, {3.0}, ACL_FLOAT, ACL_FLOAT});
//   RunCase({"TC19_scalar_i32", {}, 0, {5.0}, ACL_INT32, ACL_INT32, false, false, false, 0.0, 0.0, true});

//   RunCase({"TC20_cube_true_f32", {12800, 512}, 1, Fill(12800 * 512, 1.0), ACL_FLOAT, ACL_FLOAT, false, false, false,
//            1.0, 1e-3});
//   RunCase({"TC21_cube_true_fp16", {12800, 512}, -1, Fill(12800 * 512, 1.0), ACL_FLOAT, ACL_FLOAT16, false, false,
//            false, 2.0, 2e-2});
//   RunCase({"TC22_cube_false_small_batch", {128, 512}, 1, Fill(128 * 512, 1.0), ACL_FLOAT, ACL_FLOAT});
//   RunCase({"TC23_cube_false_small_ch", {12800, 64}, 1, Fill(12800 * 64, 1.0), ACL_FLOAT, ACL_FLOAT});
//   RunCase({"TC24_cube_false_not_last_dim", {128, 512}, 0, Fill(128 * 512, 1.0), ACL_FLOAT, ACL_FLOAT});

//   RunCase({"TC25_v2_ff", {64}, 0, Seq(64), ACL_FLOAT, ACL_FLOAT, false, false, true});
//   RunCase({"TC26_v2_tf", {64}, 0, Seq(64), ACL_FLOAT, ACL_FLOAT, true, false, true});
//   RunCase({"TC27_v2_ft", {64}, 0, Seq(64), ACL_FLOAT, ACL_FLOAT, false, true, true});
//   RunCase({"TC28_v2_tt", {64}, 0, Seq(64), ACL_FLOAT, ACL_FLOAT, true, true, true});
//   RunCase({"TC29_v2_dim_neg", {2, 16}, -1, Seq(32), ACL_FLOAT, ACL_FLOAT, true, false, true});
//   RunCase({"TC30_v2_i32", {2, 16}, 1, Seq(32), ACL_INT32, ACL_INT32, false, true, true, 0.0, 0.0, true});
//   RunCase({"TC31_v2_i64", {2, 16}, 1, Seq(32), ACL_INT64, ACL_INT64, true, true, true, 0.0, 0.0, true});
//   RunCase({"TC32_v2_fp16", {128}, 0, Seq(128), ACL_FLOAT16, ACL_FLOAT16, true, false, true, 1.0, 1e-1});
//   RunCase({"TC33_v2_bf16", {128}, 0, Seq(128), ACL_BF16, ACL_BF16, false, true, true, 2.0, 2e-1});
//   RunCase({"TC34_v2_empty", {0}, 0, {}, ACL_FLOAT, ACL_FLOAT, false, false, true});

//   RunCase({"TC35_long_1e4", {10000}, 0, Fill(10000, 1.0), ACL_FLOAT, ACL_FLOAT, false, false, false, 0.2, 1e-3, false,
//            true});
//   {
//     std::vector<double> mixed(200);
//     for (int i = 0; i < 200; ++i) mixed[static_cast<size_t>(i)] = (i % 2 == 0) ? 1e8 : 1e-6;
//     RunCase({"TC36_mixed_mag", {200}, 0, mixed, ACL_FLOAT, ACL_FLOAT, false, false, false, 1e4, 1e-2, false, true});
//   }
//   RunCase({"TC37_fp16_vs_f32_ref", {512}, 0, Fill(512, 0.1), ACL_FLOAT, ACL_FLOAT16, false, false, false, 20.0, 1e-1,
//            false, true});

//   {
//     void *oA = nullptr, *sA = nullptr;
//     aclTensor *oT = nullptr, *sT = nullptr;
//     CreateTensor(Fill(8, 0.0), {8}, ACL_FLOAT, &oA, &oT);
//     RunErrV1("TC38_err_null_self", nullptr, 0, ACL_FLOAT, oT);
//     DestroyTensor(oT, oA);

//     CreateTensor(Seq(8), {8}, ACL_FLOAT, &sA, &sT);
//     RunErrV1("TC39_err_null_out", sT, 0, ACL_FLOAT, nullptr);
//     DestroyTensor(sT, sA);
//   }

//   {
//     void *sA = nullptr, *oA = nullptr;
//     aclTensor *sT = nullptr, *oT = nullptr;
//     CreateTensor(Seq(8), {8}, ACL_FLOAT, &sA, &sT);
//     CreateTensor(Fill(8, 0.0), {8}, ACL_BOOL, &oA, &oT);
//     RunErrV1("TC40_err_out_bool", sT, 0, ACL_BOOL, oT);
//     DestroyTensor(sT, sA);
//     DestroyTensor(oT, oA);
//   }

//   {
//     void *sA = nullptr, *oA = nullptr;
//     aclTensor *sT = nullptr, *oT = nullptr;
//     CreateTensor(Seq(8), {2, 4}, ACL_FLOAT, &sA, &sT);
//     CreateTensor(Fill(8, 0.0), {4, 2}, ACL_FLOAT, &oA, &oT);
//     RunErrV1("TC41_err_shape_mismatch", sT, 0, ACL_FLOAT, oT);
//     RunErrV1("TC42_err_dim_oor", sT, 9, ACL_FLOAT, sT);
//     RunErrV1("TC43_err_dim_neg_oor", sT, -9, ACL_FLOAT, sT);
//     DestroyTensor(sT, sA);
//     DestroyTensor(oT, oA);
//   }

//   {
//     std::vector<int64_t> s9(9, 1);
//     void *sA = nullptr, *oA = nullptr;
//     aclTensor *sT = nullptr, *oT = nullptr;
//     CreateTensor(Fill(1, 1.0), s9, ACL_FLOAT, &sA, &sT);
//     CreateTensor(Fill(1, 0.0), s9, ACL_FLOAT, &oA, &oT);
//     RunErrV1("TC44_err_over_max_dim", sT, 0, ACL_FLOAT, oT);
//     DestroyTensor(sT, sA);
//     DestroyTensor(oT, oA);
//   }

//   {
//     void *sA = nullptr, *oA = nullptr;
//     aclTensor *sT = nullptr, *oT = nullptr;
//     CreateTensor(Seq(8), {8}, ACL_FLOAT, &sA, &sT);
//     CreateTensor(Fill(8, 0.0), {8}, ACL_FLOAT, &oA, &oT);
//     RunErrV2("TC45_v2_err_null_self", nullptr, 0, false, false, oT);
//     RunErrV2("TC46_v2_err_null_out", sT, 0, false, false, nullptr);
//     DestroyTensor(sT, sA);
//     DestroyTensor(oT, oA);
//   }

//   {
//     void *sA = nullptr, *oA = nullptr;
//     aclTensor *sT = nullptr, *oT = nullptr;
//     CreateTensor(Fill(8, 1.0), {8}, ACL_BOOL, &sA, &sT);
//     CreateTensor(Fill(8, 0.0), {8}, ACL_BOOL, &oA, &oT);
//     RunErrV2("TC47_v2_err_bool_unsupported", sT, 0, false, false, oT);
//     DestroyTensor(sT, sA);
//     DestroyTensor(oT, oA);
//   }

//   {
//     void *sA = nullptr, *oA = nullptr;
//     aclTensor *sT = nullptr, *oT = nullptr;
//     CreateTensor(Seq(8), {8}, ACL_FLOAT, &sA, &sT);
//     CreateTensor(Fill(8, 0.0), {8}, ACL_FLOAT16, &oA, &oT);
//     RunErrV2("TC48_v2_err_dtype_not_same", sT, 0, false, false, oT);
//     DestroyTensor(sT, sA);
//     DestroyTensor(oT, oA);
//   }

//   RunCase({"TC49_aicpu_double_v1", {64}, 0, Seq(64), ACL_FLOAT, ACL_DOUBLE, false, false, false, 1e-10, 1e-10});
//   RunCase({"TC50_aicpu_double_v2", {64}, 0, Seq(64), ACL_DOUBLE, ACL_DOUBLE, true, true, true, 1e-10, 1e-10});
//   RunCase({"TC51_aicpu_int16_v1", {64}, 0, Fill(64, 1.0), ACL_INT16, ACL_INT16, false, false, false, 0.0, 0.0, true});
//   RunCase({"TC52_aicpu_int16_v2", {64}, 0, Fill(64, 1.0), ACL_INT16, ACL_INT16, true, false, true, 0.0, 0.0, true});

//   LOG_PRINT("\nSummary: %d passed, %d failed\n", gPass, gFail);
//   Finalize();
//   return (gFail > 0) ? 1 : 0;
// }



/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * ╔═══════════════════════════════════════════════════════════════════════════════╗
 * ║ Comprehensive test for aclnnCumsum / aclnnCumsumV2                           ║
 * ║                                                                               ║
 * ║ Coverage targets:                                                             ║
 * ║  op_api/aclnn_cumsum.cpp                                                      ║
 * ║    • CheckNotNull            – null self, null out                            ║
 * ║    • CheckDtypeValid         – out unsupported (BOOL), dtype≠out              ║
 * ║    • CheckDtypeValidWithoutDtype                                               ║
 * ║        – self unsupported, out unsupported, self≠out dtype                    ║
 * ║    • CheckDim                – 0-dim→1, dim too large, too negative           ║
 * ║    • CheckShape              – shape mismatch, >8 dims                        ║
 * ║    • CheckShapeIsSupport     – 0-dim→false, dim<0 normalise,                  ║
 * ║                                dim≠last→false, batchNum<12800, ch<512, →true  ║
 * ║    • CheckCubeSupport        – dtype false, shape false, →true                ║
 * ║    • IsEmpty() early return                                                   ║
 * ║    • dim==0→DT_INT64  vs  dim!=0→DT_INT32 dimTensor                          ║
 * ║                                                                               ║
 * ║  op_api/cumsum.cpp                                                            ║
 * ║    • IsAiCoreSupport → true  (REGBASE list: FLOAT/FP16/BF16/INT32/INT64/…)   ║
 * ║    • IsAiCoreSupport → false (DOUBLE/INT16 absent from REGBASE → AiCpu)      ║
 * ║    • CumsumAiCore  exclusive×reverse four combos                              ║
 * ║    • CumsumAiCpu   exclusive×reverse four combos (via DOUBLE/INT16)           ║
 * ║                                                                               ║
 * ║  op_host/arch35/cumsum_tiling.cpp                                             ║
 * ║    • dtype → TilingCumsumAscendc (float/fp16/bf16)                           ║
 * ║    • dtype → TilingCumsum4Int    (int32/int64/int8/uint8)                     ║
 * ║                                                                               ║
 * ║  op_host/arch35/cumsum_tiling_ascendc_arch35.cpp   (float tiling)            ║
 * ║    • TILING_KEY_ONEWAY          – NGreaterCl, R full, M≥coreNum              ║
 * ║    • TILING_KEY_ONEWAY borrow-N – NGreaterCl, R full, M<coreNum               ║
 * ║    • TILING_KEY_ONEWAY RN>UB   – NGreaterCl, R full, but R×N > UB            ║
 * ║    • TILING_KEY_UB_SS_ONEWAY   – NGreaterCl, R not full, M provides cores    ║
 * ║    • TILING_KEY_CORE_SS_ONEWAY – NGreaterCl, R not full, borrow R, ONEWAY    ║
 * ║    • TILING_KEY_TWOWAY         – NLesserCl, R×N≥clSize, R large (fold≥2)     ║
 * ║    • TILING_KEY_CORE_SS_TWOWAY – NLesserCl, borrow R, TWOWAY sklansky        ║
 * ║    • MRNGreaterCl / MRNLesserCl – all dims small                              ║
 * ║    • dtCast_ path (fp16/bf16)                                                 ║
 * ║                                                                               ║
 * ║  op_host/arch35/cumsum_tiling_ascendc_int_arch35.cpp   (int tiling)           ║
 * ║    • GetInputDims axis=0, axis=last, axis=middle (3-way)                     ║
 * ║    • GetAxisLpUnit  TD_rightA / TD_leftA / TD_R                               ║
 * ║    • CalcTilingKey  CUM_NO_SPLIT / CUM_WITH_GROUP / CUM_AR_SPLIT              ║
 * ║    • dtypeSize=1 (INT8/UINT8) → vlSize_ halved                               ║
 * ╚═══════════════════════════════════════════════════════════════════════════════╝
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <climits>
#include "acl/acl.h"
#include "aclnnop/aclnn_cumsum.h"

// ============================================================
// Macros
// ============================================================
#define CHECK_RET(cond, return_expr) \
    do { if (!(cond)) { return_expr; } } while (0)
#define LOG_PRINT(msg, ...) do { printf(msg, ##__VA_ARGS__); } while (0)

// ============================================================
// Global state
// ============================================================
static int         g_pass   = 0;
static int         g_fail   = 0;
static aclrtStream g_stream = nullptr;
static const int32_t kDeviceId = 0;

static void Pass(const char *name) { LOG_PRINT("[PASS] %s\n", name); g_pass++; }
static void Fail(const char *name, const char *why = "") {
    LOG_PRINT("[FAIL] %s  (%s)\n", name, why); g_fail++;
}

// ============================================================
// ACL lifecycle
// ============================================================
static int AclSetup() {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed %d\n", ret); return ret);
    ret = aclrtSetDevice(kDeviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed %d\n", ret); return ret);
    ret = aclrtCreateStream(&g_stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed %d\n", ret); return ret);
    return 0;
}
static void AclTeardown() {
    aclrtDestroyStream(g_stream);
    aclrtResetDevice(kDeviceId);
    aclFinalize();
}

// ============================================================
// FP16 / BF16 conversion
// ============================================================
static uint16_t F32ToFP16(float f) {
    union { float f; uint32_t u; } v = {f};
    uint32_t b = v.u;
    uint16_t sign = (uint16_t)((b >> 16u) & 0x8000u);
    int exp = (int)((b >> 23u) & 0xffu) - 127 + 15;
    uint32_t mant = b & 0x7fffffu;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00u;
    return sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}
static float FP16ToF32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    if (!exp && !mant) return 0.f;
    if (exp == 31) {
        union { uint32_t u; float f; } v = {sign | 0x7f800000u | (mant << 13)};
        return v.f;
    }
    if (!exp) { while (!(mant & 0x400u)) { mant <<= 1; exp--; } exp++; mant &= 0x3ffu; }
    union { uint32_t u; float f; } v = {sign | ((exp + 112u) << 23) | (mant << 13)};
    return v.f;
}
static uint16_t F32ToBF16(float f) {
    union { float f; uint32_t u; } v = {f};
    return (uint16_t)(v.u >> 16);
}
static float BF16ToF32(uint16_t b) {
    union { uint32_t u; float f; } v = {(uint32_t)b << 16};
    return v.f;
}

// ============================================================
// Dtype utilities
// ============================================================
static size_t DtypeBytes(aclDataType dt) {
    switch (dt) {
        case ACL_FLOAT:   return 4;
        case ACL_FLOAT16: return 2;
        case ACL_BF16:    return 2;
        case ACL_INT32:   return 4;
        case ACL_INT64:   return 8;
        case ACL_INT8:    return 1;
        case ACL_UINT8:   return 1;
        case ACL_DOUBLE:  return 8;
        case ACL_INT16:   return 2;
        case ACL_BOOL:    return 1;
        default:          return 4;
    }
}

static std::vector<uint8_t> PackData(const std::vector<double> &src, aclDataType dt) {
    size_t n = src.size(), bsz = DtypeBytes(dt);
    std::vector<uint8_t> out(n * bsz);
    for (size_t i = 0; i < n; i++) {
        void *p = out.data() + i * bsz;
        switch (dt) {
            case ACL_FLOAT:   { float    v=(float)src[i];  memcpy(p,&v,4); break; }
            case ACL_FLOAT16: { uint16_t v=F32ToFP16((float)src[i]); memcpy(p,&v,2); break; }
            case ACL_BF16:    { uint16_t v=F32ToBF16((float)src[i]); memcpy(p,&v,2); break; }
            case ACL_INT32:   { int32_t  v=(int32_t)src[i]; memcpy(p,&v,4); break; }
            case ACL_INT64:   { int64_t  v=(int64_t)src[i]; memcpy(p,&v,8); break; }
            case ACL_INT8:    { int8_t   v=(int8_t)src[i];  memcpy(p,&v,1); break; }
            case ACL_UINT8:   { uint8_t  v=(uint8_t)src[i]; memcpy(p,&v,1); break; }
            case ACL_DOUBLE:  { double   v=src[i];           memcpy(p,&v,8); break; }
            case ACL_INT16:   { int16_t  v=(int16_t)src[i]; memcpy(p,&v,2); break; }
            case ACL_BOOL:    { uint8_t  v=(src[i]!=0)?1:0; memcpy(p,&v,1); break; }
            default: break;
        }
    }
    return out;
}

static std::vector<double> UnpackData(void *devAddr, int64_t n, aclDataType dt) {
    size_t bsz = DtypeBytes(dt);
    std::vector<uint8_t> buf((size_t)n * bsz);
    aclrtMemcpy(buf.data(), buf.size(), devAddr, buf.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<double> out((size_t)n);
    for (int64_t i = 0; i < n; i++) {
        const void *p = buf.data() + (size_t)i * bsz;
        switch (dt) {
            case ACL_FLOAT:   { float    v; memcpy(&v,p,4); out[i]=v; break; }
            case ACL_FLOAT16: { uint16_t v; memcpy(&v,p,2); out[i]=FP16ToF32(v); break; }
            case ACL_BF16:    { uint16_t v; memcpy(&v,p,2); out[i]=BF16ToF32(v); break; }
            case ACL_INT32:   { int32_t  v; memcpy(&v,p,4); out[i]=v; break; }
            case ACL_INT64:   { int64_t  v; memcpy(&v,p,8); out[i]=(double)v; break; }
            case ACL_INT8:    { int8_t   v; memcpy(&v,p,1); out[i]=v; break; }
            case ACL_UINT8:   { uint8_t  v; memcpy(&v,p,1); out[i]=v; break; }
            case ACL_DOUBLE:  { double   v; memcpy(&v,p,8); out[i]=v; break; }
            case ACL_INT16:   { int16_t  v; memcpy(&v,p,2); out[i]=v; break; }
            case ACL_BOOL:    { uint8_t  v; memcpy(&v,p,1); out[i]=v; break; }
            default: break;
        }
    }
    return out;
}

// ============================================================
// Tensor helpers
// ============================================================
static int64_t ShapeProd(const std::vector<int64_t> &s) {
    int64_t n = 1; for (auto d : s) n *= d; return n;
}

static int CreateTensor(const std::vector<double> &data,
                         const std::vector<int64_t> &shape,
                         aclDataType dt,
                         void **devAddr, aclTensor **tensor) {
    int64_t n     = ShapeProd(shape);
    size_t  bytes = (size_t)n * DtypeBytes(dt);

    if (bytes == 0) {
        *devAddr = nullptr;
    } else {
        auto packed = PackData(data, dt);
        auto ret = aclrtMalloc(devAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclrtMalloc(%zu) failed %d\n", bytes, ret); return ret);
        ret = aclrtMemcpy(*devAddr, bytes, packed.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    if (shape.empty()) {
        *tensor = aclCreateTensor(nullptr, 0, dt, nullptr, 0,
                                   aclFormat::ACL_FORMAT_ND, nullptr, 0, *devAddr);
    } else {
        std::vector<int64_t> strides(shape.size(), 1);
        for (int i = (int)shape.size() - 2; i >= 0; i--)
            strides[i] = shape[i + 1] * strides[i + 1];
        *tensor = aclCreateTensor(shape.data(), shape.size(), dt,
                                   strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                   shape.data(), shape.size(), *devAddr);
    }
    CHECK_RET(*tensor != nullptr,
              LOG_PRINT("aclCreateTensor returned null\n"); return -1);
    return 0;
}

static void DestroyTensor(aclTensor *t, void *addr) {
    if (t)    aclDestroyTensor(t);
    if (addr) aclrtFree(addr);
}

// ============================================================
// CPU reference: N-dim cumsum with exclusive / reverse
// ============================================================
static void CpuCumsum(const std::vector<double> &in,
                       std::vector<double> &out,
                       const std::vector<int64_t> &shape,
                       int64_t dim,
                       bool exclusive = false,
                       bool reverse   = false) {
    int     nd = (int)shape.size();
    int64_t N  = ShapeProd(shape);
    out.assign((size_t)N, 0.0);
    if (N == 0) return;
    if (nd == 0) { out = in; return; }
    if (dim < 0) dim += nd;

    int64_t outer = 1, dimSz = shape[dim], inner = 1;
    for (int i = 0; i < dim; i++)        outer *= shape[i];
    for (int i = dim + 1; i < nd; i++)   inner *= shape[i];

    for (int64_t o = 0; o < outer; o++) {
        for (int64_t n = 0; n < inner; n++) {
            int64_t base = o * dimSz * inner + n;
            double  sum  = 0.0;
            auto step = [&](int64_t k) {
                int64_t idx = base + k * inner;
                if (exclusive) { out[idx] = sum; sum += in[idx]; }
                else           { sum += in[idx]; out[idx] = sum; }
            };
            if (!reverse) { for (int64_t k = 0;        k < dimSz; k++) step(k); }
            else          { for (int64_t k = dimSz - 1; k >= 0;   k--) step(k); }
        }
    }
}

// ============================================================
// Comparison helpers
// ============================================================
static bool CompareFloat(const char *name,
                          const std::vector<double> &actual,
                          const std::vector<double> &expected,
                          double atol, double rtol,
                          bool printStats = false) {
    double  maxErr = 0.0;
    int64_t maxIdx = -1;
    bool    ok     = true;
    for (int64_t i = 0; i < (int64_t)actual.size(); i++) {
        double err = std::abs(actual[i] - expected[i]);
        double tol = atol + rtol * std::abs(expected[i]);
        if (err > maxErr) { maxErr = err; maxIdx = i; }
        if (err > tol)    ok = false;
    }
    if (printStats || !ok) {
        LOG_PRINT("  Max error: %e at pos %lld\n", maxErr, (long long)maxIdx);
        if (!actual.empty())
            LOG_PRINT("  Expected[last]=%.8f  Actual[last]=%.8f\n",
                       expected.back(), actual.back());
    }
    return ok;
}

static bool CompareExact(const std::vector<double> &actual,
                          const std::vector<double> &expected) {
    for (size_t i = 0; i < actual.size(); i++)
        if ((int64_t)actual[i] != (int64_t)expected[i]) return false;
    return true;
}

// ============================================================
// Test config + runner
// ============================================================
struct TC {
    const char          *name;
    std::vector<int64_t>  shape;
    int64_t               dim       = 0;
    std::vector<double>   input;
    aclDataType           selfDt    = ACL_FLOAT;
    aclDataType           outDt     = ACL_FLOAT;
    bool                  exclusive = false;
    bool                  reverse   = false;
    bool                  useV2     = false;
    double                atol      = 1e-5;
    double                rtol      = 1e-5;
    bool                  isInt     = false;
    bool                  printStats = false;
};

static bool RunTC(const TC &tc) {
    int64_t n = ShapeProd(tc.shape);

    void *selfAddr = nullptr, *outAddr = nullptr;
    aclTensor *self = nullptr, *out = nullptr;

    if (CreateTensor(tc.input, tc.shape, tc.selfDt, &selfAddr, &self) != 0) {
        Fail(tc.name, "CreateSelf failed"); return false;
    }
    std::vector<double> outInit((size_t)n, 0.0);
    if (CreateTensor(outInit, tc.shape, tc.outDt, &outAddr, &out) != 0) {
        DestroyTensor(self, selfAddr);
        Fail(tc.name, "CreateOut failed"); return false;
    }

    uint64_t wsSize = 0; aclOpExecutor *exec = nullptr;
    aclnnStatus ret;
    if (!tc.useV2)
        ret = aclnnCumsumGetWorkspaceSize(self, tc.dim, tc.outDt, out, &wsSize, &exec);
    else
        ret = aclnnCumsumV2GetWorkspaceSize(self, tc.dim, tc.exclusive, tc.reverse,
                                             out, &wsSize, &exec);

    if (ret != ACL_SUCCESS) {
        DestroyTensor(self, selfAddr); DestroyTensor(out, outAddr);
        char buf[64]; snprintf(buf, sizeof(buf), "GetWSSize failed: %d", (int)ret);
        Fail(tc.name, buf); return false;
    }

    void *wsAddr = nullptr;
    if (wsSize > 0 &&
        aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        DestroyTensor(self, selfAddr); DestroyTensor(out, outAddr);
        Fail(tc.name, "ws malloc failed"); return false;
    }

    if (!tc.useV2) ret = (aclnnStatus)aclnnCumsum(wsAddr,  wsSize, exec, g_stream);
    else           ret = (aclnnStatus)aclnnCumsumV2(wsAddr, wsSize, exec, g_stream);
    aclrtSynchronizeStream(g_stream);

    bool ok = false;
    if (ret != ACL_SUCCESS) {
        char buf[64]; snprintf(buf, sizeof(buf), "exec failed: %d", (int)ret);
        Fail(tc.name, buf);
    } else if (n == 0) {
        Pass(tc.name); ok = true;   // empty tensor – IsEmpty() early return
    } else {
        auto actual = UnpackData(outAddr, n, tc.outDt);
        std::vector<double> expected;
        CpuCumsum(tc.input, expected, tc.shape, tc.dim, tc.exclusive, tc.reverse);
        if (tc.isInt) ok = CompareExact(actual, expected);
        else          ok = CompareFloat(tc.name, actual, expected,
                                         tc.atol, tc.rtol, tc.printStats);
        if (ok) Pass(tc.name);
        else    Fail(tc.name, "result mismatch");
    }

    DestroyTensor(self, selfAddr); DestroyTensor(out, outAddr);
    if (wsAddr) aclrtFree(wsAddr);
    return ok;
}

// Error-path runner: expects GetWorkspaceSize to fail
static bool RunErrV1(const char *name,
                     aclTensor *self, int64_t dim,
                     aclDataType dtype, aclTensor *out) {
    uint64_t ws = 0; aclOpExecutor *exec = nullptr;
    auto ret = aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, &ws, &exec);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[PASS] %s  (correctly rejected, ret=%d)\n", name, (int)ret);
        g_pass++; return true;
    }
    LOG_PRINT("[FAIL] %s  (expected failure but got success)\n", name);
    g_fail++; return false;
}
static bool RunErrV2(const char *name,
                     aclTensor *self, int64_t dim,
                     bool excl, bool rev, aclTensor *out) {
    uint64_t ws = 0; aclOpExecutor *exec = nullptr;
    auto ret = aclnnCumsumV2GetWorkspaceSize(self, dim, excl, rev, out, &ws, &exec);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[PASS] %s  (correctly rejected, ret=%d)\n", name, (int)ret);
        g_pass++; return true;
    }
    LOG_PRINT("[FAIL] %s  (expected failure but got success)\n", name);
    g_fail++; return false;
}

// Input generators
static std::vector<double> Seq(int64_t n, double start = 1.0) {
    std::vector<double> v((size_t)n);
    std::iota(v.begin(), v.end(), start);
    return v;
}
static std::vector<double> Fill(int64_t n, double val) {
    return std::vector<double>((size_t)n, val);
}

// ============================================================
// MAIN
// ============================================================
int main() {
    if (AclSetup() != 0) { LOG_PRINT("ACL setup failed\n"); return -1; }
    LOG_PRINT("========= Cumsum Comprehensive Tests =========\n\n");

    // ──────────────────────────────────────────────────────────
    // GROUP 1 – V1 dtype coverage  (covers all 7 AiCore dtypes)
    // ──────────────────────────────────────────────────────────
    LOG_PRINT("--- Group 1: V1 dtype coverage ---\n");
    RunTC({"TC01_f32_dim0",  {4}, 0, Seq(4),    ACL_FLOAT,   ACL_FLOAT});
    RunTC({"TC02_f32_2d_d1", {2,4}, 1, Seq(8),  ACL_FLOAT,   ACL_FLOAT});
    RunTC({"TC03_fp16_out",  {4}, 0, Seq(4),    ACL_FLOAT,   ACL_FLOAT16, false,false,false,5e-3,5e-3});
    RunTC({"TC04_bf16_out",  {4}, 0, Seq(4),    ACL_FLOAT,   ACL_BF16,    false,false,false,1e-2,1e-2});
    RunTC({"TC05_int32",     {4}, 0, Seq(4),    ACL_INT32,   ACL_INT32,   false,false,false,0.,0.,true});
    RunTC({"TC06_int64",     {4}, 0, Seq(4),    ACL_INT64,   ACL_INT64,   false,false,false,0.,0.,true});
    RunTC({"TC07_int8",      {4}, 0, {1,2,3,4}, ACL_INT8,    ACL_INT8,    false,false,false,0.,0.,true});
    RunTC({"TC08_uint8",     {4}, 0, {1,2,3,4}, ACL_UINT8,   ACL_UINT8,   false,false,false,0.,0.,true});

    // ──────────────────────────────────────────────────────────
    // GROUP 2 – V1 dimension coverage
    // ──────────────────────────────────────────────────────────
    LOG_PRINT("\n--- Group 2: V1 dimension coverage ---\n");
    RunTC({"TC09_3d_dim0",  {2,3,4}, 0,  Seq(24), ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC10_3d_dim1",  {2,3,4}, 1,  Seq(24), ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC11_3d_dim2",  {2,3,4}, 2,  Seq(24), ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC12_neg_m1",   {2,4},  -1,  Seq(8),  ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC13_neg_m2",   {2,4},  -2,  Seq(8),  ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC14_4d_dim3",  {2,2,2,4}, 3, Seq(32), ACL_FLOAT, ACL_FLOAT});

    // ──────────────────────────────────────────────────────────
    // GROUP 3 – V1 special values
    // ──────────────────────────────────────────────────────────
    LOG_PRINT("\n--- Group 3: V1 special values ---\n");
    RunTC({"TC15_zeros",     {4}, 0, Fill(4,0.),         ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC16_negatives", {4}, 0, {-1.,-2.,-3.,-4.},  ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC17_alt_pm",    {6}, 0, {1.,-1.,1.,-1.,1.,-1.}, ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC18_i32_zeros", {4}, 0, Fill(4,0.), ACL_INT32, ACL_INT32, false,false,false,0.,0.,true});
    RunTC({"TC19_single",    {1}, 0, {42.},       ACL_FLOAT, ACL_FLOAT});

    // ──────────────────────────────────────────────────────────
    // GROUP 4 – Empty tensor and 0-dim scalar
    // ──────────────────────────────────────────────────────────
    LOG_PRINT("\n--- Group 4: Empty / scalar ---\n");
    RunTC({"TC20_empty_0x4", {0,4}, 0, {},    ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC21_empty_2x0", {2,0}, 0, {},    ACL_INT32, ACL_INT32});
    RunTC({"TC22_scalar_f32",{}, 0, {5.0},   ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC23_scalar_i32",{}, 0, {3.0},   ACL_INT32, ACL_INT32, false,false,false,0.,0.,true});

    // ──────────────────────────────────────────────────────────
    // GROUP 5 – CheckCubeSupport branches
    // ──────────────────────────────────────────────────────────
    LOG_PRINT("\n--- Group 5: CubeSupport branches ---\n");
    {   // CubeSupport=TRUE – f32
        int64_t B=12800, C=512;
        RunTC({"TC24_cube_f32", {B,C}, 1, Fill(B*C,1.0),
               ACL_FLOAT,ACL_FLOAT,false,false,false,1e-3,1e-4});
    }
    {   // CubeSupport=TRUE – fp16, dim=-1 (covers dim<0 in CheckShapeIsSupport)
        int64_t B=12800, C=512;
        RunTC({"TC25_cube_fp16_negdim",{B,C},-1,Fill(B*C,1.0),
               ACL_FLOAT,ACL_FLOAT16,false,false,false,1.0,1e-2});
    }
    {   // CubeSupport=TRUE – bf16
        int64_t B=12800, C=512;
        RunTC({"TC26_cube_bf16",{B,C},1,Fill(B*C,1.0),
               ACL_FLOAT,ACL_BF16,false,false,false,2.0,1e-2});
    }
    {   // CubeSupport=FALSE – INT32 dtype
        RunTC({"TC27_cube_int32_no",{128,512},1,Fill(128*512,1.0),
               ACL_INT32,ACL_INT32,false,false,false,0.,0.,true});
    }
    {   // CubeSupport=FALSE – batchNum<12800
        RunTC({"TC28_cube_small_batch",{128,512},1,Fill(128*512,1.0),
               ACL_FLOAT,ACL_FLOAT});
    }
    {   // CubeSupport=FALSE – channelNum<512
        int64_t B=12800, C=64;
        RunTC({"TC29_cube_small_ch",{B,C},1,Seq(B*C),ACL_FLOAT,ACL_FLOAT});
    }
    {   // CubeSupport=FALSE – dim≠last
        RunTC({"TC30_cube_not_lastdim",{128,512},0,Fill(128*512,1.0),
               ACL_FLOAT,ACL_FLOAT,false,false,false,1e-4,1e-4});
    }
    {   // CubeSupport=FALSE – 1D with dim=0 (batchNum=1)
        RunTC({"TC31_cube_1d_false",{128},0,Seq(128),ACL_FLOAT,ACL_FLOAT});
    }

    // ──────────────────────────────────────────────────────────
    // GROUP 6 – V1 error paths
    // ──────────────────────────────────────────────────────────
    LOG_PRINT("\n--- Group 6: V1 error paths ---\n");
    {   // null self
        void *oA=nullptr; aclTensor *oT=nullptr;
        CreateTensor(Fill(4,0.),{4},ACL_FLOAT,&oA,&oT);
        RunErrV1("TC32_null_self", nullptr, 0, ACL_FLOAT, oT);
        DestroyTensor(oT,oA);
    }
    {   // null out
        void *sA=nullptr; aclTensor *sT=nullptr;
        CreateTensor(Seq(4),{4},ACL_FLOAT,&sA,&sT);
        RunErrV1("TC33_null_out", sT, 0, ACL_FLOAT, nullptr);
        DestroyTensor(sT,sA);
    }
    {   // out has BOOL dtype → OP_CHECK_DTYPE_NOT_SUPPORT(out,…)
        void *sA=nullptr,*oA=nullptr; aclTensor *sT=nullptr,*oT=nullptr;
        CreateTensor(Seq(4),    {4},ACL_FLOAT,&sA,&sT);
        CreateTensor(Fill(4,0.),{4},ACL_BOOL, &oA,&oT);
        RunErrV1("TC34_out_bool_unsupported", sT, 0, ACL_BOOL, oT);
        DestroyTensor(sT,sA); DestroyTensor(oT,oA);
    }
    {   // dtype param ≠ out dtype → OP_CHECK_DTYPE_NOT_MATCH
        void *sA=nullptr,*oA=nullptr; aclTensor *sT=nullptr,*oT=nullptr;
        CreateTensor(Seq(4),    {4},ACL_FLOAT,  &sA,&sT);
        CreateTensor(Fill(4,0.),{4},ACL_FLOAT,  &oA,&oT);
        RunErrV1("TC35_dtype_mismatch", sT, 0, ACL_FLOAT16, oT);
        DestroyTensor(sT,sA); DestroyTensor(oT,oA);
    }
    {   // shape mismatch
        void *sA=nullptr,*oA=nullptr; aclTensor *sT=nullptr,*oT=nullptr;
        CreateTensor(Seq(6),    {2,3},ACL_FLOAT,&sA,&sT);
        CreateTensor(Fill(6,0.),{3,2},ACL_FLOAT,&oA,&oT);
        RunErrV1("TC36_shape_mismatch", sT, 0, ACL_FLOAT, oT);
        DestroyTensor(sT,sA); DestroyTensor(oT,oA);
    }
    {   // 9-dimensional tensor → OP_CHECK_MAX_DIM(8)
        std::vector<int64_t> s9(9,1LL);
        void *sA=nullptr,*oA=nullptr; aclTensor *sT=nullptr,*oT=nullptr;
        CreateTensor(Fill(1,1.),s9,ACL_FLOAT,&sA,&sT);
        CreateTensor(Fill(1,0.),s9,ACL_FLOAT,&oA,&oT);
        RunErrV1("TC37_9dim", sT, 0, ACL_FLOAT, oT);
        DestroyTensor(sT,sA); DestroyTensor(oT,oA);
    }
    {   // dim too large
        void *sA=nullptr,*oA=nullptr; aclTensor *sT=nullptr,*oT=nullptr;
        CreateTensor(Seq(8),    {2,4},ACL_FLOAT,&sA,&sT);
        CreateTensor(Fill(8,0.),{2,4},ACL_FLOAT,&oA,&oT);
        RunErrV1("TC38_dim_too_large", sT, 5, ACL_FLOAT, oT);
        DestroyTensor(sT,sA); DestroyTensor(oT,oA);
    }
    {   // dim too negative
        void *sA=nullptr,*oA=nullptr; aclTensor *sT=nullptr,*oT=nullptr;
        CreateTensor(Seq(8),    {2,4},ACL_FLOAT,&sA,&sT);
        CreateTensor(Fill(8,0.),{2,4},ACL_FLOAT,&oA,&oT);
        RunErrV1("TC39_dim_too_neg", sT, -5, ACL_FLOAT, oT);
        DestroyTensor(sT,sA); DestroyTensor(oT,oA);
    }

    // ──────────────────────────────────────────────────────────
    // GROUP 7 – V2 functional: all 4 exclusive×reverse combos
    // ──────────────────────────────────────────────────────────
    LOG_PRINT("\n--- Group 7: V2 exclusive/reverse ---\n");
    RunTC({"TC40_v2_ff",{6},0,Seq(6),ACL_FLOAT,ACL_FLOAT,false,false,true});
    RunTC({"TC41_v2_tf",{6},0,Seq(6),ACL_FLOAT,ACL_FLOAT,true, false,true});
    RunTC({"TC42_v2_ft",{6},0,Seq(6),ACL_FLOAT,ACL_FLOAT,false,true, true});
    RunTC({"TC43_v2_tt",{6},0,Seq(6),ACL_FLOAT,ACL_FLOAT,true, true, true});
    // V2 dim=0 → DT_INT64 dimTensor
    RunTC({"TC44_v2_dim0_dt64",{3,4},0,Seq(12),ACL_FLOAT,ACL_FLOAT,false,true,true});
    // V2 dim=1 → DT_INT32 dimTensor
    RunTC({"TC45_v2_dim1_dt32",{3,4},1,Seq(12),ACL_FLOAT,ACL_FLOAT,true,false,true});
    // V2 fp16 same dtype
    RunTC({"TC46_v2_fp16",{4},0,{1,2,3,4},ACL_FLOAT16,ACL_FLOAT16,true,false,true,5e-3,5e-3});
    // V2 bf16
    RunTC({"TC47_v2_bf16",{4},0,{1,2,3,4},ACL_BF16,ACL_BF16,false,true,true,1e-2,1e-2});
    // V2 int32 reverse
    RunTC({"TC48_v2_int32_rev",{2,4},1,Seq(8),ACL_INT32,ACL_INT32,false,true,true,0.,0.,true});
    // V2 int64 exclusive
    RunTC({"TC49_v2_int64_excl",{4},0,Seq(4),ACL_INT64,ACL_INT64,true,false,true,0.,0.,true});
    // V2 int8
    RunTC({"TC50_v2_int8_tt",{4},0,{1,2,3,4},ACL_INT8,ACL_INT8,true,true,true,0.,0.,true});
    // V2 uint8
    RunTC({"TC51_v2_uint8_rev",{4},0,{1,2,3,4},ACL_UINT8,ACL_UINT8,false,true,true,0.,0.,true});
    // V2 empty
    RunTC({"TC52_v2_empty",{0},0,{},ACL_FLOAT,ACL_FLOAT,false,false,true});
    // V2 3D tensor
    RunTC({"TC53_v2_3d",{2,3,4},1,Seq(24),ACL_FLOAT,ACL_FLOAT,true,true,true});
    // V2 negative dim
    RunTC({"TC54_v2_neg_dim",{2,4},-1,Seq(8),ACL_FLOAT,ACL_FLOAT,true,false,true});

    // ──────────────────────────────────────────────────────────
    // GROUP 8 – V2 error paths
    // ──────────────────────────────────────────────────────────
    LOG_PRINT("\n--- Group 8: V2 error paths ---\n");
    {   // null self
        void *oA=nullptr; aclTensor *oT=nullptr;
        CreateTensor(Fill(4,0.),{4},ACL_FLOAT,&oA,&oT);
        RunErrV2("TC55_v2_null_self",nullptr,0,false,false,oT);
        DestroyTensor(oT,oA);
    }
    {   // null out
        void *sA=nullptr; aclTensor *sT=nullptr;
        CreateTensor(Seq(4),{4},ACL_FLOAT,&sA,&sT);
        RunErrV2("TC56_v2_null_out",sT,0,false,false,nullptr);
        DestroyTensor(sT,sA);
    }
    {   // self=BOOL → OP_CHECK_DTYPE_NOT_SUPPORT(self,…)
        void *sA=nullptr,*oA=nullptr; aclTensor *sT=nullptr,*oT=nullptr;
        CreateTensor(Fill(4,0.),{4},ACL_BOOL,&sA,&sT);
        CreateTensor(Fill(4,0.),{4},ACL_BOOL,&oA,&oT);
        RunErrV2("TC57_v2_self_bool",sT,0,false,false,oT);
        DestroyTensor(sT,sA); DestroyTensor(oT,oA);
    }
    {   // self=FLOAT, out=BOOL → OP_CHECK_DTYPE_NOT_SUPPORT(out,…)
        void *sA=nullptr,*oA=nullptr; aclTensor *sT=nullptr,*oT=nullptr;
        CreateTensor(Seq(4),    {4},ACL_FLOAT,&sA,&sT);
        CreateTensor(Fill(4,0.),{4},ACL_BOOL, &oA,&oT);
        RunErrV2("TC58_v2_out_bool",sT,0,false,false,oT);
        DestroyTensor(sT,sA); DestroyTensor(oT,oA);
    }
    {   // self≠out dtype → OP_CHECK_DTYPE_NOT_SAME
        void *sA=nullptr,*oA=nullptr; aclTensor *sT=nullptr,*oT=nullptr;
        CreateTensor(Seq(4),    {4},ACL_FLOAT,   &sA,&sT);
        CreateTensor(Fill(4,0.),{4},ACL_FLOAT16, &oA,&oT);
        RunErrV2("TC59_v2_dtype_not_same",sT,0,false,false,oT);
        DestroyTensor(sT,sA); DestroyTensor(oT,oA);
    }
    {   // V2 dim out of range
        void *sA=nullptr,*oA=nullptr; aclTensor *sT=nullptr,*oT=nullptr;
        CreateTensor(Seq(4),    {4},ACL_FLOAT,&sA,&sT);
        CreateTensor(Fill(4,0.),{4},ACL_FLOAT,&oA,&oT);
        RunErrV2("TC60_v2_dim_oor",sT,5,false,false,oT);
        DestroyTensor(sT,sA); DestroyTensor(oT,oA);
    }
    {   // V2 shape mismatch
        void *sA=nullptr,*oA=nullptr; aclTensor *sT=nullptr,*oT=nullptr;
        CreateTensor(Seq(6),    {2,3},ACL_FLOAT,&sA,&sT);
        CreateTensor(Fill(6,0.),{3,2},ACL_FLOAT,&oA,&oT);
        RunErrV2("TC61_v2_shape_mismatch",sT,0,false,false,oT);
        DestroyTensor(sT,sA); DestroyTensor(oT,oA);
    }

    // ──────────────────────────────────────────────────────────
    // GROUP 9 – AiCpu dtype path
    // DOUBLE and INT16 pass API check (in ASCEND910B list) but are
    // NOT in REGBASE_DTYPE_SUPPORT_LIST → IsAiCoreSupport=false → AiCpu
    // ──────────────────────────────────────────────────────────
    LOG_PRINT("\n--- Group 9: AiCpu dtype path (DOUBLE, INT16) ---\n");
    RunTC({"TC62_double_v1",    {4}, 0,Seq(4), ACL_FLOAT,  ACL_DOUBLE, false,false,false,1e-9,1e-9});
    RunTC({"TC63_double_2d",    {2,4},1,Seq(8),ACL_FLOAT,  ACL_DOUBLE, false,false,false,1e-9,1e-9});
    // V2 DOUBLE – covers AiCpu + exclusive/reverse path in cumsum.cpp
    RunTC({"TC64_double_v2_excl",{4},0,{1,2,3,4},ACL_DOUBLE,ACL_DOUBLE,true, false,true,1e-9,1e-9});
    RunTC({"TC65_double_v2_rev", {4},0,{1,2,3,4},ACL_DOUBLE,ACL_DOUBLE,false,true, true,1e-9,1e-9});
    RunTC({"TC66_double_v2_tt",  {4},0,{1,2,3,4},ACL_DOUBLE,ACL_DOUBLE,true, true, true,1e-9,1e-9});
    RunTC({"TC67_int16_v1",      {4},0,{1,2,3,4},ACL_INT16, ACL_INT16, false,false,false,0.,0.,true});
    RunTC({"TC68_int16_v2_tt",   {4},0,{1,2,3,4},ACL_INT16, ACL_INT16, true, true, true,0.,0.,true});

    // ──────────────────────────────────────────────────────────
    // GROUP 10 – Sequence length diversity (tiling boundary tests)
    // ──────────────────────────────────────────────────────────
    LOG_PRINT("\n--- Group 10: Sequence length diversity ---\n");
    RunTC({"TC69_f32_n1",   {1},    0,{3.0},       ACL_FLOAT,ACL_FLOAT});
    RunTC({"TC70_f32_n16",  {16},   0,Seq(16),     ACL_FLOAT,ACL_FLOAT});
    RunTC({"TC71_f32_n64",  {64},   0,Seq(64),     ACL_FLOAT,ACL_FLOAT});
    RunTC({"TC72_f32_n128", {128},  0,Seq(128),    ACL_FLOAT,ACL_FLOAT});
    RunTC({"TC73_f32_n256", {256},  0,Seq(256),    ACL_FLOAT,ACL_FLOAT});
    RunTC({"TC74_f32_n100", {100},  0,Seq(100),    ACL_FLOAT,ACL_FLOAT});
    RunTC({"TC75_f32_n500", {500},  0,Fill(500,1.),ACL_FLOAT,ACL_FLOAT,false,false,false,1e-3,1e-4});
    RunTC({"TC76_f32_n1000",{1000}, 0,Fill(1000,1.),ACL_FLOAT,ACL_FLOAT,false,false,false,1e-3,1e-4});
    RunTC({"TC77_fp16_n128",{128},  0,Seq(128),    ACL_FLOAT,ACL_FLOAT16,false,false,false,1.,1e-2});
    RunTC({"TC78_i32_n256", {256},  0,Seq(256),    ACL_INT32,ACL_INT32,false,false,false,0.,0.,true});
    RunTC({"TC79_i32_n1000",{1000}, 0,Seq(1000),   ACL_INT32,ACL_INT32,false,false,false,0.,0.,true});
    RunTC({"TC80_i64_n256", {256},  0,Seq(256),    ACL_INT64,ACL_INT64,false,false,false,0.,0.,true});

    // ──────────────────────────────────────────────────────────
    // GROUP 11 – Precision analysis
    // ──────────────────────────────────────────────────────────
    LOG_PRINT("\n--- Group 11: Precision analysis ---\n");
    // TC81 Error accumulation – float32 10000×1.0
    // Theory: accumulated error ≈ n × ε₃₂ × sum ≈ 10000 × 1.19e-7 × 5000 ≈ 0.006
    LOG_PRINT("[TC81] float32 error accumulation n=10000 all-ones\n");
    LOG_PRINT("  Theory: max_err ≈ n × ε₃₂ × mean_sum ≈ 10000 × 1.19e-7 × 5000 ≈ 0.006\n");
    RunTC({"TC81_err_accum_f32",{10000},0,Fill(10000,1.0),ACL_FLOAT,ACL_FLOAT,
           false,false,false,0.1,1e-4,false,true});

    // TC82 Large-small magnitude: 1e8 swamps 1e-6 (ULP(1e8)≈8, so 1e-6 lost)
    LOG_PRINT("[TC82] float32 mixed magnitude [1e8, 1e-6, …]\n");
    {
        std::vector<double> inp(200);
        for (int i=0;i<200;i++) inp[i]=(i%2==0)?1e8:1e-6;
        RunTC({"TC82_mixed_mag",{200},0,inp,ACL_FLOAT,ACL_FLOAT,
               false,false,false,1e4,1e-2,false,true});
    }
    // TC83-84 FP16 vs float32 comparison (same input)
    LOG_PRINT("[TC83-84] fp16 vs f32, n=50 – precision gap\n");
    RunTC({"TC83_f32_n50", {50},0,Seq(50),ACL_FLOAT,ACL_FLOAT, false,false,false,1e-3,1e-5,false,true});
    RunTC({"TC84_fp16_n50",{50},0,Seq(50),ACL_FLOAT,ACL_FLOAT16,false,false,false,20.,5e-2,false,true});
    // TC85 Non-representable 0.1 × 200 (float32)
    LOG_PRINT("[TC85] float32 cumsum of 0.1×200\n");
    RunTC({"TC85_0p1_accum",{200},0,Fill(200,0.1),ACL_FLOAT,ACL_FLOAT,
           false,false,false,1e-3,1e-4,false,true});
    // TC86 bf16 vs float32
    RunTC({"TC86_bf16_n50",{50},0,Seq(50),ACL_FLOAT,ACL_BF16,false,false,false,30.,5e-2,false,true});
    // TC87 Alternating +/-1 cancellation
    {
        std::vector<double> inp(1000);
        for (int i=0;i<1000;i++) inp[i]=(i%2==0)?1.:-1.;
        RunTC({"TC87_alt_cancel",{1000},0,inp,ACL_FLOAT,ACL_FLOAT,false,false,false,1e-4,1e-5});
    }
    // TC88 V2 exclusive precision
    LOG_PRINT("[TC88] V2 exclusive n=500\n");
    RunTC({"TC88_v2_excl_prec",{500},0,Fill(500,1.0),ACL_FLOAT,ACL_FLOAT,
           true,false,true,1e-3,1e-4,false,true});

    // ──────────────────────────────────────────────────────────
    // GROUP 12 – Float tiling key coverage
    // Shapes chosen to trigger each of the 8 TILING_KEY_* values
    // in cumsum_tiling_ascendc_arch35.cpp
    // Hardware: 910_93, clSize=64, blockSize=32, ubSize≈192KB,
    //           vRegSize=256, coreNum≈20
    // ──────────────────────────────────────────────────────────
    LOG_PRINT("\n--- Group 12: Float tiling key coverage ---\n");

    // TILING_KEY_ONEWAY (1001): N≥16 (NGreaterCl), R fits in UB (R×clSize≤ubSize),
    // M≥coreNum → standard core split, R×N also fits UB
    // lenM=100, lenR=100, lenN=16: lenN×4=64 (NGreaterCl); R×64=6400≤192K; M=100≥20
    RunTC({"TC89_tiling_oneway", {100,100,16}, 1, Fill(100*100*16,1.0),
           ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-3, 1e-4});

    // TILING_KEY_ONEWAY with M<coreNum → borrow-N branch:
    // lenM=10<20, lenR=100, lenN=16; borrows N axis for core utilisation
    RunTC({"TC90_tiling_borrow_n", {10,100,16}, 1, Fill(10*100*16,1.0),
           ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-3, 1e-4});

    // TILING_KEY_ONEWAY with R×N > UB → "N not fully loadable" branch:
    // lenM=64≥20, lenR=3000 (3000×64=192K, borderline), lenN=32; R×N×4=384K>192K
    RunTC({"TC91_tiling_rn_nofit", {64,3000,32}, 1, Fill(64*3000*32,1.0),
           ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-2, 1e-4});

    // TILING_KEY_UB_SS_ONEWAY (1011): N large, R not full load (R×64>192K),
    // M≥coreNum/2 → M split + UB sklansky scan
    // lenM=50≥10, lenR=4000 (4000×64=256K>192K), lenN=16
    RunTC({"TC92_tiling_ub_ss_oneway", {50,4000,16}, 1, Fill(50*4000*16,1.0),
           ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-2, 1e-4});

    // TILING_KEY_CORE_SS_ONEWAY (1101): N large, R not full load, M too small
    // → borrow R across cores, ONEWAY sklansky on borrowed R slices
    // lenM=1, lenR=4000, lenN=32; borrows R (coreNum=20, each handles R/20=200)
    RunTC({"TC93_tiling_core_ss_oneway", {1,4000,32}, 1, Fill(1*4000*32,1.0),
           ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-2, 1e-3});

    // TILING_KEY_TWOWAY (1002): N<16 (NLesserCl), R large enough for fold≥2,
    // total UB fits → TWOWAY sklansky (two-direction scan)
    // lenM=1, lenR=512, lenN=1; alignN=32, foldCount=2 → TWOWAY
    RunTC({"TC94_tiling_twoway_512", {512}, 0, Fill(512,1.0),
           ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-3, 1e-4});

    // TWOWAY with larger R (higher foldCount)
    RunTC({"TC95_tiling_twoway_1024", {1024}, 0, Fill(1024,1.0),
           ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-3, 1e-4});

    // TILING_KEY_CORE_SS_TWOWAY (1102): N small, borrow R, TWOWAY per slice
    // lenM=1, lenR=8000, lenN=2; borrowR=20; per-slice R=400, foldCount=2 → TWOWAY
    RunTC({"TC96_tiling_core_ss_twoway", {1,8000,2}, 1, Fill(1*8000*2,1.0),
           ACL_FLOAT, ACL_FLOAT, false, false, false, 1.0, 1e-3});

    // MRNLesserCl path (all dims tiny, M×R×N×4 < clSize=64)
    RunTC({"TC97_tiling_mrn_lesscl", {2,2,2}, 1, Seq(8),
           ACL_FLOAT, ACL_FLOAT});

    // MRNGreaterCl path (M×R×N×4 ≥ 64 but R×N×4 < 64 and N×4 < 64)
    // lenM=100, lenR=4, lenN=2; M×R×N×4=3200≥64; R×N×4=32<64; N×4=8<64
    RunTC({"TC98_tiling_mrngreater", {100,4,2}, 2, Seq(100*4*2),
           ACL_FLOAT, ACL_FLOAT});

    // fp16 dtCast_ path in NGreaterCl (clNSize=192 for fp16)
    RunTC({"TC99_tiling_fp16_dtcast", {64,100,16}, 1, Fill(64*100*16,1.0),
           ACL_FLOAT, ACL_FLOAT16, false, false, false, 1.0, 1e-2});

    // fp16 UB_SS path: lenR×192 > ubSize → R not full load for fp16
    // lenM=50, lenR=1200(×192=230K>192K), lenN=16
    RunTC({"TC100_tiling_fp16_ub_ss", {50,1200,16}, 1, Fill(50*1200*16,1.0),
           ACL_FLOAT, ACL_FLOAT16, false, false, false, 2.0, 1e-2});

    // bf16 also exercises dtCast_ path (same as fp16)
    RunTC({"TC101_tiling_bf16_ngr", {64,100,16}, 1, Fill(64*100*16,1.0),
           ACL_FLOAT, ACL_BF16, false, false, false, 2.0, 1e-2});

    // ──────────────────────────────────────────────────────────
    // GROUP 13 – Int tiling coverage
    // Targets GetInputDims axis branches, GetAxisLpUnit TD paths,
    // CalcTilingKey variants, and vlSize_ halving for INT8/UINT8
    // ──────────────────────────────────────────────────────────
    LOG_PRINT("\n--- Group 13: Int tiling coverage ---\n");

    // ── axis=0 branch in GetInputDims ──
    // INT32 [100, 100] dim=0: axis=0, rightAxisLen=100
    //   rightAxisLen×4=400 > vlSize_/2=128 → TD_rightA
    //   rightAxisLen×4=400 ≥ vlSize_=256 → CUM_NO_SPLIT
    RunTC({"TC102_i32_axis0_nosplit", {100,100}, 0, Fill(100*100,1.0),
           ACL_INT32, ACL_INT32, false, false, false, 0., 0., true});

    // INT32 [100, 10] dim=0: axis=0, rightAxisLen=10
    //   rightAxisLen×4=40 ≤ 128 → NOT TD_rightA
    //   leftAxisLen=1 ≤ coreNum/2 → NOT TD_leftA → TD_R
    //   rightAxisLen×4=40 < 256 → CUM_AR_SPLIT (or CUM_WITH_GROUP)
    RunTC({"TC103_i32_axis0_tdr", {100,10}, 0, Fill(100*10,1.0),
           ACL_INT32, ACL_INT32, false, false, false, 0., 0., true});

    // ── axis=last branch ──
    // INT32 [50, 100] dim=1: axis=last, leftAxisLen=50 > coreNum/CORE_GATE(~10)
    //   → TD_leftA; rightAxisLen=1 → CUM_AR_SPLIT
    RunTC({"TC104_i32_axislast_tdla", {50,100}, 1, Fill(50*100,1.0),
           ACL_INT32, ACL_INT32, false, false, false, 0., 0., true});

    // INT32 [5, 100] dim=1: axis=last, leftAxisLen=5 ≤ 10 → NOT TD_leftA → TD_R
    RunTC({"TC105_i32_axislast_tdr", {5,100}, 1, Fill(500,1.0),
           ACL_INT32, ACL_INT32, false, false, false, 0., 0., true});

    // ── axis=middle branch ──
    // INT32 [50, 100, 32] dim=1: axis=middle, leftAxisLen=50>10 (TD_leftA),
    //   rightAxisLen=32×4=128 ≤ vlSize_/2=128 → NOT TD_rightA
    //   CUM_NO_SPLIT: rightAxisLen×4=128<256 → NO; CUM_AR_SPLIT
    RunTC({"TC106_i32_axismid_tdla", {50,100,32}, 1, Fill(50*100*32,1.0),
           ACL_INT32, ACL_INT32, false, false, false, 0., 0., true});

    // INT32 [50, 100, 64] dim=1: axis=middle, leftAxisLen=50>10,
    //   rightAxisLen=64×4=256 ≥ vlSize_=256 → CUM_NO_SPLIT
    RunTC({"TC107_i32_axismid_nosplit", {50,100,64}, 1, Fill(50*100*64,1.0),
           ACL_INT32, ACL_INT32, false, false, false, 0., 0., true});

    // ── CUM_WITH_GROUP: R-axis dominates core split ──
    // [1, 1280, 1] INT32 dim=1: leftAxisLen=1, midAxisLen=1280, rightAxisLen=1
    //   NOT TD_rightA, NOT TD_leftA → TD_R
    //   With comLeftA=64, rLpUnit≈1280, rLpCnt=1 → typically LA or RA split.
    //   Try a shape where R split weight wins:
    // [1, 20, 1] INT32 dim=1 small tensor – R split likely winner
    RunTC({"TC108_i32_cum_with_group", {1,20,1}, 1, Fill(20,1.0),
           ACL_INT32, ACL_INT32, false, false, false, 0., 0., true});

    // ── INT64 axis and tiling ──
    // [100, 200] INT64 dim=0: rightAxisLen=200, 200×8=1600 > vlSize_/2=128 → TD_rightA
    //   CUM_NO_SPLIT: 200×8=1600 ≥ 256 → CUM_NO_SPLIT
    RunTC({"TC109_i64_axis0_nosplit", {100,200}, 0, Fill(100*200,1.0),
           ACL_INT64, ACL_INT64, false, false, false, 0., 0., true});

    // [50, 100] INT64 dim=1: axis=last, leftAxisLen=50>10 → TD_leftA
    RunTC({"TC110_i64_axislast_tdla", {50,100}, 1, Fill(50*100,1.0),
           ACL_INT64, ACL_INT64, false, false, false, 0., 0., true});

    // ── INT8: vlSize_ halved to 128; dtypeSize=1 ──
    // TD_rightA for INT8: rightAxisLen×1 > vlSize_/2=64 → rightAxisLen > 64
    // [200, 65] INT8 dim=0: axis=0, rightAxisLen=65 > 64 → TD_rightA
    //   CUM_NO_SPLIT: 65 < 128 → NO; CUM_AR_SPLIT or CUM_WITH_GROUP
    RunTC({"TC111_i8_axis0_tdr_a", {200,65}, 0, Fill(200*65,1.0),
           ACL_INT8, ACL_INT8, false, false, false, 0., 0., true});

    // [200, 130] INT8 dim=0: rightAxisLen=130 ≥ 128 → CUM_NO_SPLIT
    RunTC({"TC112_i8_axis0_nosplit", {200,130}, 0, Fill(200*130,1.0),
           ACL_INT8, ACL_INT8, false, false, false, 0., 0., true});

    // [200, 10] INT8 dim=0: rightAxisLen=10 ≤ 64 → NOT TD_rightA
    //   leftAxisLen=1 ≤ 10 → TD_R
    RunTC({"TC113_i8_axis0_tdr_r", {200,10}, 0, Fill(200*10,1.0),
           ACL_INT8, ACL_INT8, false, false, false, 0., 0., true});

    // [50, 100] INT8 dim=1: axis=last, leftAxisLen=50>10 → TD_leftA
    RunTC({"TC114_i8_axislast_tdla", {50,100}, 1, Fill(50*100,1.0),
           ACL_INT8, ACL_INT8, false, false, false, 0., 0., true});

    // ── UINT8 ──
    // [100, 80] UINT8 dim=0: rightAxisLen=80 > 64 → TD_rightA; 80<128 → CUM_AR_SPLIT
    RunTC({"TC115_u8_axis0_tdra", {100,80}, 0, Fill(100*80,1.0),
           ACL_UINT8, ACL_UINT8, false, false, false, 0., 0., true});

    // [100, 130] UINT8 dim=0: rightAxisLen=130 ≥ 128 → CUM_NO_SPLIT
    RunTC({"TC116_u8_axis0_nosplit", {100,130}, 0, Fill(100*130,1.0),
           ACL_UINT8, ACL_UINT8, false, false, false, 0., 0., true});

    // ── Int tiling with V2 flags (exclusive/reverse affect kernel params) ──
    RunTC({"TC117_i32_v2_excl", {100,50}, 1, Fill(100*50,1.0),
           ACL_INT32, ACL_INT32, true, false, true, 0., 0., true});
    RunTC({"TC118_i32_v2_rev",  {100,50}, 1, Fill(100*50,1.0),
           ACL_INT32, ACL_INT32, false,true, true, 0., 0., true});
    RunTC({"TC119_i64_v2_tt",   {50,100}, 1, Seq(50*100),
           ACL_INT64, ACL_INT64, true, true, true, 0., 0., true});
    RunTC({"TC120_i8_v2_excl",  {50,100}, 1, Fill(50*100,1.0),
           ACL_INT8,  ACL_INT8,  true, false,true, 0., 0., true});

    // ──────────────────────────────────────────────────────────
    // GROUP 14 – Extra V2 and mixed shapes
    // ──────────────────────────────────────────────────────────
    LOG_PRINT("\n--- Group 14: Extra shapes ---\n");
    RunTC({"TC121_1x1000_d1",  {1,1000},  1,Seq(1000),  ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC122_1000x1_d0",  {1000,1},  0,Seq(1000),  ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC123_f32_n127",   {127},     0,Seq(127),   ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC124_f32_n129",   {129},     0,Seq(129),   ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC125_f32_n512",   {512},     0,Seq(512),   ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC126_i32_n127",   {127},     0,Seq(127),   ACL_INT32, ACL_INT32,false,false,false,0.,0.,true});
    RunTC({"TC127_i32_n1024",  {1024},    0,Seq(1024),  ACL_INT32, ACL_INT32,false,false,false,0.,0.,true});
    RunTC({"TC128_v2_3d_tt",   {2,3,5},   2,Seq(30),    ACL_FLOAT, ACL_FLOAT,true,true,true});
    RunTC({"TC129_v2_i64_tt",  {200},     0,Seq(200),   ACL_INT64, ACL_INT64,true,true,true,0.,0.,true});
    RunTC({"TC130_v2_fp16_rev",{2,50},    1,Seq(100),   ACL_FLOAT16,ACL_FLOAT16,false,true,true,1.,1e-1});
    RunTC({"TC131_v2_i32_1000",{1000},    0,Seq(1000),  ACL_INT32, ACL_INT32,true,false,true,0.,0.,true});
    RunTC({"TC132_f32_100x10", {100,10},  1,Seq(1000),  ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC133_f32_10x100", {10,100},  0,Seq(1000),  ACL_FLOAT, ACL_FLOAT});
    RunTC({"TC134_fp16_2d",    {4,32},    1,Seq(128),   ACL_FLOAT, ACL_FLOAT16,false,false,false,1.,1e-2});
    RunTC({"TC135_bf16_2d",    {4,32},    1,Seq(128),   ACL_FLOAT, ACL_BF16,   false,false,false,5.,1e-1});

    // ──────────────────────────────────────────────────────────
    // Summary
    // ──────────────────────────────────────────────────────────
    LOG_PRINT("\n================================================\n");
    LOG_PRINT("Summary: %d passed, %d failed\n", g_pass, g_fail);
    LOG_PRINT("================================================\n");

    AclTeardown();
    return (g_fail > 0) ? 1 : 0;
}