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
#include <random>
#include <unordered_set>
#include <vector>

namespace aclco::test {

template <typename T>
struct ContainsTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    std::vector<T> queryKeys;
    DeviceBuffer<T> dKeys;
    DeviceBuffer<unsigned char> dResult;
    std::optional<roaring_bitmap_factory::RoaringBitmapT<T>> bitmap;
};

template <typename T>
ContainsTestContext<T>& GetContext() {
    static ContainsTestContext<T> ctx;
    return ctx;
}

template <typename T>
void SetupContainsTest(std::string bitmapFile, long long numQueryKeys, int seed) {
    auto& ctx = GetContext<T>();
    ctx.stream = ctx.streamGuard.stream;
    using Key = T;

    // 从 tests/testdata/ 加载二进制测试数据文件
    std::ifstream file("tests/testdata/" + bitmapFile, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open bitmap file: tests/testdata/" + bitmapFile);
    }
    auto fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<std::byte> serialized(static_cast<std::size_t>(fileSize));
    file.read(reinterpret_cast<char*>(serialized.data()), fileSize);

    ctx.bitmap = roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, ctx.stream);
    Sync(ctx.stream);

    // 随机唯一查询key (distribution::unique)
    std::mt19937_64 rng(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<std::uint64_t> dist(
        0, static_cast<std::uint64_t>(std::numeric_limits<Key>::max()));
    std::unordered_set<Key> seen;
    ctx.queryKeys.reserve(static_cast<std::size_t>(numQueryKeys));
    while (ctx.queryKeys.size() < static_cast<std::size_t>(numQueryKeys)) {
        Key v = static_cast<Key>(dist(rng));
        if (seen.insert(v).second) {
            ctx.queryKeys.push_back(v);
        }
    }

    ctx.dKeys = DeviceBuffer<T>(ctx.queryKeys.size());
    ctx.dKeys.CopyFromHostAsync(ctx.queryKeys.data(), ctx.queryKeys.size(), ctx.stream);
    ctx.dResult = DeviceBuffer<unsigned char>(ctx.queryKeys.size());
}

template <typename T>
TestResult TestContains() {
    auto& ctx = GetContext<T>();

    auto start = std::chrono::high_resolution_clock::now();
    ctx.bitmap->Contains(static_cast<void*>(ctx.dKeys.Data()),
                         static_cast<void*>(ctx.dResult.Data()),
                         aclco::Extent<std::size_t>(ctx.queryKeys.size()), ctx.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

// uint32: 2^32 queries, uint64: 2^31 queries, 使用二进制测试数据文件
REGISTER_PERFORMANCE_TEST(containsCase1, (TestContains<uint32_t>), (SetupContainsTest<uint32_t>), std::string, long long, int);
REGISTER_PERFORMANCE_TEST(containsCase2, (TestContains<uint64_t>), (SetupContainsTest<uint64_t>), std::string, long long, int);

REGISTER_PERFORMANCE_ARGS(containsCase1, "contains_uint32",
    (std::initializer_list<std::tuple<std::string,long long,int>>{
        {"bitmapwithruns.bin", 1LL << 32, 200},
        {"bitmapwithruns.bin", 80000000, 200}
    }), std::string, long long, int);

REGISTER_PERFORMANCE_ARGS(containsCase2, "contains_uint64",
    (std::initializer_list<std::tuple<std::string,long long,int>>{
        {"portable_bitmap64.bin", 1LL << 31, 200},
        {"portable_bitmap64.bin", 80000000, 200}
    }), std::string, long long, int);

}
