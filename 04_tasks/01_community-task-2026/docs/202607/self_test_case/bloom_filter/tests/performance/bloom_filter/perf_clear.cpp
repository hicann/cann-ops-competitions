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
struct ClearTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    std::optional<bloom_filter_factory::BloomFilterT<Key>> filter;
};

template <typename Key, typename Word, int BlockBits, int PatternBits, int HorizontalLayout, int VerticalLayout>
ClearTestContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>& GetContext() {
    static ClearTestContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout> ctx;
    return ctx;
}

template <typename Key, typename Word, int BlockBits, int PatternBits, int HorizontalLayout, int VerticalLayout>
void SetupClearTest(int filterSizeMB) {
    auto& ctx = GetContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>();
    ctx.stream = ctx.streamGuard.stream;

    // filterSizeMB -> numBlocks
    constexpr int bitsOfWord = static_cast<int>(sizeof(Word) * 8);
    constexpr int wordsPerBlock = BlockBits / bitsOfWord;
    constexpr int bytesPerBlock = static_cast<int>(sizeof(Word)) * wordsPerBlock;
    std::size_t numBlocks = static_cast<std::size_t>(filterSizeMB) * 1024 * 1024 / bytesPerBlock;
    ctx.filter = bloom_filter_factory::MakeBloomFilter<Key>(numBlocks, ctx.stream);
}

template <typename Key, typename Word, int BlockBits, int PatternBits, int HorizontalLayout, int VerticalLayout>
TestResult TestClear() {
    auto& ctx = GetContext<Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>();

    auto start = std::chrono::high_resolution_clock::now();
    ctx.filter->Clear(ctx.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

// 默认基准测试: <Key, Word, BlockBits, PatternBits, HorizontalLayout, VerticalLayout>
REGISTER_PERFORMANCE_TEST(clearCase1, (TestClear<int32_t, uint32_t, 256, 8, 8, 1>), (SetupClearTest<int32_t, uint32_t, 256, 8, 8, 1>), int);
REGISTER_PERFORMANCE_TEST(clearCase2, (TestClear<int64_t, uint32_t, 256, 8, 8, 1>), (SetupClearTest<int64_t, uint32_t, 256, 8, 8, 1>), int);

REGISTER_PERFORMANCE_ARGS(clearCase1, "clear_int32",
    (std::initializer_list<std::tuple<int>>{{32}, {256}, {2048}}), int);
REGISTER_PERFORMANCE_ARGS(clearCase2, "clear_int64",
    (std::initializer_list<std::tuple<int>>{{32}, {256}, {2048}}), int);

}
