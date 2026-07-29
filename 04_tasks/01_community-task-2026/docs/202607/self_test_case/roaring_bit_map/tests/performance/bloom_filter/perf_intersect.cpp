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
struct IntersectTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    int numKeys;
    std::vector<Key> hostKeys;
    DeviceBuffer<Key> dKeys;
    std::optional<bloom_filter_factory::BloomFilterT<Key>> filterA;
    std::optional<bloom_filter_factory::BloomFilterT<Key>> filterB;
};

template <typename Key, typename Word, int BlockBits, int PatternBits, int HorizontalLayout, int VerticalLayout>
IntersectTestContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>& GetContext() {
    static IntersectTestContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout> ctx;
    return ctx;
}

template <typename Key, typename Word, int BlockBits, int PatternBits, int HorizontalLayout, int VerticalLayout>
void SetupIntersectTest(int numKeys, int filterSizeMB) {
    auto& ctx = GetContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>();
    ctx.stream = ctx.streamGuard.stream;
    ctx.numKeys = numKeys;

    // 顺序key, filterSizeMB -> numBlocks
    ctx.hostKeys = bloom_filter_factory::GenerateKeys<Key>(numKeys, "sequential");

    constexpr int bitsOfWord = static_cast<int>(sizeof(Word) * 8);
    constexpr int wordsPerBlock = BlockBits / bitsOfWord;
    constexpr int bytesPerBlock = static_cast<int>(sizeof(Word)) * wordsPerBlock;
    std::size_t numBlocks = static_cast<std::size_t>(filterSizeMB) * 1024 * 1024 / bytesPerBlock;

    ctx.dKeys = DeviceBuffer<Key>(ctx.hostKeys.size());
    ctx.dKeys.CopyFromHostAsync(ctx.hostKeys.data(), ctx.hostKeys.size(), ctx.stream);

    ctx.filterA = bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, ctx.stream);
    ctx.filterB = bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, ctx.stream);

    // filterB 预填充, 不计入计时区域
    ctx.filterB->Add(static_cast<void*>(ctx.dKeys.Data()),
                     aclco::Extent<std::size_t>(ctx.hostKeys.size()), ctx.stream);
}

template <typename Key, typename Word, int BlockBits, int PatternBits, int HorizontalLayout, int VerticalLayout>
TestResult TestIntersect() {
    auto& ctx = GetContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>();

    // 每次迭代复位 filterA (不计时)
    ctx.filterA->Clear(ctx.stream);
    ctx.filterA->Add(static_cast<void*>(ctx.dKeys.Data()),
                     aclco::Extent<std::size_t>(ctx.hostKeys.size()), ctx.stream);

    auto start = std::chrono::high_resolution_clock::now();
    ctx.filterA->Intersect(*ctx.filterB, ctx.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

// 默认基准测试: <Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>
REGISTER_PERFORMANCE_TEST(intersectCase1, (TestIntersect<int32_t, uint32_t, 256, 8, 1, 8>), (SetupIntersectTest<int32_t, uint32_t, 256, 8, 1, 8>), int, int);
REGISTER_PERFORMANCE_TEST(intersectCase2, (TestIntersect<int64_t, uint32_t, 256, 8, 1, 8>), (SetupIntersectTest<int64_t, uint32_t, 256, 8, 1, 8>), int, int);

REGISTER_PERFORMANCE_ARGS(intersectCase1, "intersect_int32",
    (std::initializer_list<std::tuple<int,int>>{
        {80000000, 32}, {80000000, 256}, {80000000, 2048}
    }), int, int);
REGISTER_PERFORMANCE_ARGS(intersectCase2, "intersect_int64",
    (std::initializer_list<std::tuple<int,int>>{
        {80000000, 32}, {80000000, 256}, {80000000, 2048}
    }), int, int);

}
