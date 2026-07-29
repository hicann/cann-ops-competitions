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
struct AddTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    std::size_t numKeys;
    std::vector<Key> hostKeys;
    DeviceBuffer<Key> dKeys;
    std::optional<bloom_filter_factory::BloomFilterT<Key>> filter;
};

template <typename Key, typename Word, int BlockBits, int PatternBits, int HorizontalLayout, int VerticalLayout>
AddTestContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>& GetContext() {
    static AddTestContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout> ctx;
    return ctx;
}

template <typename Key, typename Word, int BlockBits, int PatternBits, int HorizontalLayout, int VerticalLayout>
void SetupAddTest(int numKeys, int filterSizeMB) {
    auto& ctx = GetContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>();
    ctx.stream = ctx.streamGuard.stream;
    ctx.numKeys = static_cast<std::size_t>(numKeys);

    // 顺序key (从0开始的 counting_iterator)
    ctx.hostKeys = bloom_filter_factory::GenerateKeys<Key>(ctx.numKeys, "sequential");

    // filterSizeMB -> numBlocks
    constexpr int bitsOfWord = static_cast<int>(sizeof(Word) * 8);
    constexpr int wordsPerBlock = BlockBits / bitsOfWord;
    constexpr int bytesPerBlock = static_cast<int>(sizeof(Word)) * wordsPerBlock;
    std::size_t numBlocks = static_cast<std::size_t>(filterSizeMB) * 1024 * 1024 / bytesPerBlock;

    ctx.dKeys = DeviceBuffer<Key>(ctx.hostKeys.size());
    ctx.dKeys.CopyFromHostAsync(ctx.hostKeys.data(), ctx.hostKeys.size(), ctx.stream);

    ctx.filter = bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, ctx.stream);
}

template <typename Key, typename Word, int BlockBits, int PatternBits, int HorizontalLayout, int VerticalLayout>
TestResult TestAdd() {
    auto& ctx = GetContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>();

    ctx.filter->Clear(ctx.stream);

    auto start = std::chrono::high_resolution_clock::now();
    ctx.filter->Add(static_cast<void*>(ctx.dKeys.Data()),
                    aclco::Extent<std::size_t>(ctx.numKeys), ctx.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

// 默认基准测试: <Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>
REGISTER_PERFORMANCE_TEST(addCase1, (TestAdd<int32_t, uint32_t, 256, 8, 8, 1>), (SetupAddTest<int32_t, uint32_t, 256, 8, 8, 1>), int, int);
REGISTER_PERFORMANCE_TEST(addCase2, (TestAdd<int64_t, uint32_t, 256, 8, 8, 1>), (SetupAddTest<int64_t, uint32_t, 256, 8, 8, 1>), int, int);

REGISTER_PERFORMANCE_ARGS(addCase1, "add_int32",
    (std::initializer_list<std::tuple<int,int>>{
        {80000000, 32}, {80000000, 256}, {80000000, 2048}
    }), int, int);
REGISTER_PERFORMANCE_ARGS(addCase2, "add_int64",
    (std::initializer_list<std::tuple<int,int>>{
        {80000000, 32}, {80000000, 256}, {80000000, 2048}
    }), int, int);

}
