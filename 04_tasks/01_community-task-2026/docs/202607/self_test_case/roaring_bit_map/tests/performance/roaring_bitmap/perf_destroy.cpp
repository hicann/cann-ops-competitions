/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../performance_test_framework.h"
#include "roaring_bitmap_factory.h"

#include <fstream>
#include <memory>
#include <vector>

namespace aclco::test {

template <typename T>
struct DestroyTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    std::vector<std::byte> serializedData;
};

template <typename T>
DestroyTestContext<T>& GetContext() {
    static DestroyTestContext<T> ctx;
    return ctx;
}

template <typename T>
void SetupDestroyTest(std::string bitmapFile, long long /*numInputs*/) {
    auto& ctx = GetContext<T>();
    ctx.stream = ctx.streamGuard.stream;

    // 从 tests/testdata/ 加载二进制测试数据文件
    std::ifstream file("tests/testdata/" + bitmapFile, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open bitmap file: tests/testdata/" + bitmapFile);
    }
    auto fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    ctx.serializedData.resize(static_cast<std::size_t>(fileSize));
    file.read(reinterpret_cast<char*>(ctx.serializedData.data()), fileSize);
}

template <typename T>
TestResult TestDestroy() {
    auto& ctx = GetContext<T>();
    using Key = T;

    using RB = roaring_bitmap_factory::RoaringBitmapT<Key>;

    // 每次迭代创建 bitmap 用于销毁 (对齐上游：measure 循环内 create -> destroy -> recreate)
    auto bitmapPtr = std::make_unique<RB>(
        roaring_bitmap_factory::MakeRoaringBitmap<Key>(ctx.serializedData, ctx.stream));
    Sync(ctx.stream);

    auto start = std::chrono::high_resolution_clock::now();
    bitmapPtr.reset();
    Sync(ctx.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

// uint32: bitmapwithruns.bin + NumInputs=2^32, uint64: portable_bitmap64.bin + NumInputs=2^31
REGISTER_PERFORMANCE_TEST(destroyCase1, (TestDestroy<uint32_t>), (SetupDestroyTest<uint32_t>), std::string, long long);
REGISTER_PERFORMANCE_TEST(destroyCase2, (TestDestroy<uint64_t>), (SetupDestroyTest<uint64_t>), std::string, long long);

REGISTER_PERFORMANCE_ARGS(destroyCase1, "destroy_uint32",
    (std::initializer_list<std::tuple<std::string,long long>>{
        {"bitmapwithruns.bin", 1LL << 32},
        {"bitmapwithruns.bin", 80000000}
    }),
    std::string, long long);

REGISTER_PERFORMANCE_ARGS(destroyCase2, "destroy_uint64",
    (std::initializer_list<std::tuple<std::string,long long>>{
        {"portable_bitmap64.bin", 1LL << 31},
        {"portable_bitmap64.bin", 80000000}
    }),
    std::string, long long);
}