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
#include "bloom_filter_factory.h"

namespace aclco::test {

template <typename Key, typename Word, int BlockBits, int PatternBits, int HorizontalLayout, int VerticalLayout>
struct ContainsTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    std::size_t numKeys;
    int filterSizeMB;
    std::vector<Key> queryKeys;
    std::vector<Key> buildKeys;
    DeviceBuffer<Key> dQueryKeys;
    DeviceBuffer<Key> dBuildKeys;
    DeviceBuffer<unsigned char> dResult;
    std::optional<bloom_filter_factory::BloomFilterT<Key>> filter;
};

template <typename Key, typename Word, int BlockBits, int PatternBits, int HorizontalLayout, int VerticalLayout>
ContainsTestContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>& GetContext() {
    static ContainsTestContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout> ctx;
    return ctx;
}

template <typename Key, typename Word, int BlockBits, int PatternBits, int HorizontalLayout, int VerticalLayout>
void SetupContainsTest(int numKeys, int filterSizeMB) {
    auto& ctx = GetContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>();
    ctx.stream = ctx.streamGuard.stream;
    ctx.numKeys = static_cast<std::size_t>(numKeys);
    ctx.filterSizeMB = filterSizeMB;

    // filterSizeMB -> numBlocks
    constexpr int bitsOfWord = static_cast<int>(sizeof(Word) * 8);
    constexpr int wordsPerBlock = BlockBits / bitsOfWord;
    constexpr int bytesPerBlock = static_cast<int>(sizeof(Word)) * wordsPerBlock;
    std::size_t numBlocks = static_cast<std::size_t>(filterSizeMB) * 1024 * 1024 / bytesPerBlock;

    // FPR-optimal 数量的 key 用于预填充
    auto numBuildKeys = static_cast<std::size_t>(
        (static_cast<std::int64_t>(filterSizeMB) * 1024 * 1024 * 8) / (2 * PatternBits));

    // 查询 key: 0, 1, 2, ..., numKeys-1
    ctx.queryKeys = bloom_filter_factory::GenerateKeys<Key>(ctx.numKeys, "sequential");

    // 预填充 key: 0, 1, 2, ..., numBuildKeys-1 — 与查询 key 不同批次
    ctx.buildKeys = bloom_filter_factory::GenerateKeys<Key>(numBuildKeys, "sequential");

    ctx.dQueryKeys = DeviceBuffer<Key>(ctx.queryKeys.size());
    ctx.dQueryKeys.CopyFromHostAsync(ctx.queryKeys.data(), ctx.queryKeys.size(), ctx.stream);

    ctx.dBuildKeys = DeviceBuffer<Key>(ctx.buildKeys.size());
    ctx.dBuildKeys.CopyFromHostAsync(ctx.buildKeys.data(), ctx.buildKeys.size(), ctx.stream);

    ctx.dResult = DeviceBuffer<unsigned char>(ctx.queryKeys.size());

    ctx.filter = bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, ctx.stream);

    // 用 FPR-optimal 数量的 key 预填充 filter
    ctx.filter->Add(static_cast<void*>(ctx.dBuildKeys.Data()),
                    aclco::Extent<std::size_t>(ctx.buildKeys.size()), ctx.stream);
}

template <typename Key, typename Word, int BlockBits, int PatternBits, int HorizontalLayout, int VerticalLayout>
TestResult TestContains() {
    auto& ctx = GetContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>();

    auto start = std::chrono::high_resolution_clock::now();
    ctx.filter->Contains(static_cast<void*>(ctx.dQueryKeys.Data()),
                         static_cast<void*>(ctx.dResult.Data()),
                         aclco::Extent<std::size_t>(ctx.numKeys), ctx.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

// 默认基准测试: <Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>
REGISTER_PERFORMANCE_TEST(containsCase1, (TestContains<int32_t, uint32_t, 256, 8, 1, 8>), (SetupContainsTest<int32_t, uint32_t, 256, 8, 1, 8>), int, int);
REGISTER_PERFORMANCE_TEST(containsCase2, (TestContains<int64_t, uint32_t, 256, 8, 1, 8>), (SetupContainsTest<int64_t, uint32_t, 256, 8, 1, 8>), int, int);

REGISTER_PERFORMANCE_ARGS(containsCase1, "contains_int32",
    (std::initializer_list<std::tuple<int,int>>{
        {80000000, 32}, {80000000, 256}, {80000000, 2048}
    }), int, int);

REGISTER_PERFORMANCE_ARGS(containsCase2, "contains_int64",
    (std::initializer_list<std::tuple<int,int>>{
        {80000000, 32}, {80000000, 256}, {80000000, 2048}
    }), int, int);

}
