/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "tests/common/acl_env.h"
#include "tests/common/roaring_bitmap_factory.h"
#include "tests/common/device_buffer.h"
#include "tests/common/test_print.h"

TEMPLATE_TEST_CASE(
  "roaring_bitmap Contains exact match (sync)",
  "[roaring_bitmap][contains]",
  std::uint32_t,
  std::uint64_t)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  auto numElements = GENERATE(1u, 128u, 4096u, 65536u, 100000u);
  auto numQueryKeys = GENERATE(1u, 128u, 4096u, 65536u);
  auto pattern = GENERATE(as<std::string>{}, "in-set", "out-of-set", "mixed");

  std::string params = "numElements=" + std::to_string(numElements) +
                       ", numQueryKeys=" + std::to_string(numQueryKeys) +
                       ", pattern=" + pattern;
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Contains exact match (sync)", Key, 0, 0, numQueryKeys, params);

  auto elements = aclco::test::roaring_bitmap_factory::GenerateElements<Key>(numElements);
  auto serialized = aclco::test::roaring_bitmap_factory::Serialize<Key>(elements);
  auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
  aclco::test::Sync(stream);

  auto queryKeys = aclco::test::roaring_bitmap_factory::GenerateQueryKeys<Key>(
      numQueryKeys, pattern, elements);

  std::unordered_set<Key> groundTruth(elements.begin(), elements.end());

  SECTION("contains exact match, matchRate must be 1.0") {
    PRINT_SECTION("contains exact match, matchRate must be 1.0");
    auto results = aclco::test::roaring_bitmap_factory::ContainsKeys<Key>(bitmap, queryKeys, stream);
    double matchRate = aclco::test::roaring_bitmap_factory::ComputeMatchRate<Key>(
        queryKeys, results, groundTruth);
    PRINT_INFO("matchRate=" + std::to_string(matchRate) +
               " elements=" + std::to_string(numElements) +
               " queries=" + std::to_string(numQueryKeys) +
               " pattern=" + pattern);
    REQUIRE_PRINT(matchRate == 1.0);
  }
}

TEMPLATE_TEST_CASE(
  "roaring_bitmap ContainsAsync exact match",
  "[roaring_bitmap][containsAsync]",
  std::uint32_t,
  std::uint64_t)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  auto numElements = GENERATE(128u, 4096u, 65536u);
  auto pattern = GENERATE(as<std::string>{}, "in-set", "out-of-set", "mixed");
  auto numQueryKeys = numElements;

  std::string params = "numElements=" + std::to_string(numElements) +
                       ", pattern=" + pattern;
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("ContainsAsync exact match", Key, 0, 0, numQueryKeys, params);

  auto elements = aclco::test::roaring_bitmap_factory::GenerateElements<Key>(numElements);
  auto serialized = aclco::test::roaring_bitmap_factory::Serialize<Key>(elements);
  auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
  aclco::test::Sync(stream);

  auto queryKeys = aclco::test::roaring_bitmap_factory::GenerateQueryKeys<Key>(
      numQueryKeys, pattern, elements);
  std::unordered_set<Key> groundTruth(elements.begin(), elements.end());

  SECTION("containsAsync exact match, matchRate must be 1.0") {
    PRINT_SECTION("containsAsync exact match, matchRate must be 1.0");
    aclco::test::DeviceBuffer<Key> dKeys(queryKeys.size());
    dKeys.CopyFromHostAsync(queryKeys.data(), queryKeys.size(), stream);
    aclco::test::DeviceBuffer<unsigned char> dResult(queryKeys.size());
    dResult.MemsetZero(stream);

    bitmap.ContainsAsync(static_cast<void*>(dKeys.Data()),
                         static_cast<void*>(dResult.Data()),
                         aclco::Extent<std::size_t>(queryKeys.size()), stream);
    auto results = dResult.CopyToHost(stream);

    double matchRate = aclco::test::roaring_bitmap_factory::ComputeMatchRate<Key>(
        queryKeys, results, groundTruth);
    PRINT_INFO("matchRate=" + std::to_string(matchRate) + " (async, must be 1.0)");
    REQUIRE_PRINT(matchRate == 1.0);
  }
}

TEMPLATE_TEST_CASE(
  "roaring_bitmap Contains boundary keys",
  "[roaring_bitmap][contains]",
  std::uint32_t,
  std::uint64_t)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  std::size_t numElements = 65536u;
  std::string params = "numElements=" + std::to_string(numElements);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Contains boundary keys", Key, 0, 0, 0, params);

  auto elements = aclco::test::roaring_bitmap_factory::GenerateElements<Key>(numElements);
  auto serialized = aclco::test::roaring_bitmap_factory::Serialize<Key>(elements);
  auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
  aclco::test::Sync(stream);

  std::unordered_set<Key> groundTruth(elements.begin(), elements.end());

  SECTION("zero key query") {
    PRINT_SECTION("zero key query");
    std::vector<Key> zeroKeys = {static_cast<Key>(0)};
    auto results = aclco::test::roaring_bitmap_factory::ContainsKeys<Key>(bitmap, zeroKeys, stream);
    double matchRate = aclco::test::roaring_bitmap_factory::ComputeMatchRate<Key>(
        zeroKeys, results, groundTruth);
    REQUIRE_PRINT(matchRate == 1.0);
  }

  SECTION("max key query") {
    PRINT_SECTION("max key query");
    std::vector<Key> maxKeys = {std::numeric_limits<Key>::max()};
    auto results = aclco::test::roaring_bitmap_factory::ContainsKeys<Key>(bitmap, maxKeys, stream);
    double matchRate = aclco::test::roaring_bitmap_factory::ComputeMatchRate<Key>(
        maxKeys, results, groundTruth);
    REQUIRE_PRINT(matchRate == 1.0);
  }

  SECTION("empty query array") {
    PRINT_SECTION("empty query array");
    auto results = aclco::test::roaring_bitmap_factory::ContainsKeys<Key>(bitmap, {}, stream);
    REQUIRE_PRINT(results.empty());
  }
}

TEMPLATE_TEST_CASE(
  "roaring_bitmap Contains empty bitmap",
  "[roaring_bitmap][contains]",
  std::uint32_t,
  std::uint64_t)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  std::size_t numQueryKeys = 128u;
  std::string params = "numQueryKeys=" + std::to_string(numQueryKeys);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Contains empty bitmap", Key, 0, 0, numQueryKeys, params);

  std::vector<Key> emptyElements;
  auto serialized = aclco::test::roaring_bitmap_factory::Serialize<Key>(emptyElements);
  auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
  aclco::test::Sync(stream);

  auto queryKeys = aclco::test::roaring_bitmap_factory::GenerateQueryKeys<Key>(
      numQueryKeys, "mixed", emptyElements);

  SECTION("empty bitmap: all queries return false") {
    PRINT_SECTION("empty bitmap: all queries return false");
    auto results = aclco::test::roaring_bitmap_factory::ContainsKeys<Key>(bitmap, queryKeys, stream);
    bool allFalse = true;
    for (auto r : results) {
      if (r != 0) {
        allFalse = false;
        break;
      }
    }
    REQUIRE_PRINT(allFalse);
  }

  SECTION("empty bitmap: Size==0, Empty==true") {
    PRINT_SECTION("empty bitmap: Size==0, Empty==true");
    REQUIRE_PRINT(bitmap.Size() == 0);
    REQUIRE_PRINT(bitmap.Empty());
  }
}

TEMPLATE_TEST_CASE(
  "roaring_bitmap Contains from testdata binary files",
  "[roaring_bitmap][contains][testdata]",
  std::uint32_t,
  std::uint64_t)
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key = TestType;

  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("Contains from testdata", Key, 0, 0, 0, "testdata");

  if constexpr (std::is_same_v<Key, std::uint32_t>) {
    SECTION("bitmapwithruns.bin: load, query, verify no crash") {
      PRINT_SECTION("bitmapwithruns.bin: load, query, verify no crash");
      std::ifstream file("tests/testdata/bitmapwithruns.bin", std::ios::binary | std::ios::ate);
      REQUIRE_PRINT(file.is_open());
      auto fileSize = static_cast<std::size_t>(file.tellg());
      file.seekg(0, std::ios::beg);
      std::vector<std::byte> serialized(fileSize);
      file.read(reinterpret_cast<char*>(serialized.data()), static_cast<std::streamsize>(fileSize));

      auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
      aclco::test::Sync(stream);
      REQUIRE_PRINT(bitmap.Data() != nullptr);
      REQUIRE_PRINT(bitmap.Size() > 0);
      REQUIRE_PRINT(bitmap.SizeBytes() == fileSize);
      REQUIRE_PRINT(!bitmap.Empty());

      auto queryKeys = aclco::test::roaring_bitmap_factory::GenerateQueryKeys<Key>(
          1024u, "mixed", std::vector<Key>{});
      auto results = aclco::test::roaring_bitmap_factory::ContainsKeys<Key>(bitmap, queryKeys, stream);
      REQUIRE_PRINT(results.size() == 1024u);
    }

    SECTION("bitmapwithoutruns.bin: load, query, verify no crash") {
      PRINT_SECTION("bitmapwithoutruns.bin: load, query, verify no crash");
      std::ifstream file("tests/testdata/bitmapwithoutruns.bin", std::ios::binary | std::ios::ate);
      REQUIRE_PRINT(file.is_open());
      auto fileSize = static_cast<std::size_t>(file.tellg());
      file.seekg(0, std::ios::beg);
      std::vector<std::byte> serialized(fileSize);
      file.read(reinterpret_cast<char*>(serialized.data()), static_cast<std::streamsize>(fileSize));

      auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
      aclco::test::Sync(stream);
      REQUIRE_PRINT(bitmap.Data() != nullptr);
      REQUIRE_PRINT(bitmap.Size() > 0);
      REQUIRE_PRINT(!bitmap.Empty());

      auto queryKeys = aclco::test::roaring_bitmap_factory::GenerateQueryKeys<Key>(
          1024u, "mixed", std::vector<Key>{});
      auto results = aclco::test::roaring_bitmap_factory::ContainsKeys<Key>(bitmap, queryKeys, stream);
      REQUIRE_PRINT(results.size() == 1024u);
    }
  }

  if constexpr (std::is_same_v<Key, std::uint64_t>) {
    SECTION("portable_bitmap64.bin: load, query, verify no crash") {
      PRINT_SECTION("portable_bitmap64.bin: load, query, verify no crash");
      std::ifstream file("tests/testdata/portable_bitmap64.bin", std::ios::binary | std::ios::ate);
      REQUIRE_PRINT(file.is_open());
      auto fileSize = static_cast<std::size_t>(file.tellg());
      file.seekg(0, std::ios::beg);
      std::vector<std::byte> serialized(fileSize);
      file.read(reinterpret_cast<char*>(serialized.data()), static_cast<std::streamsize>(fileSize));

      auto bitmap = aclco::test::roaring_bitmap_factory::MakeRoaringBitmap<Key>(serialized, stream);
      aclco::test::Sync(stream);
      REQUIRE_PRINT(bitmap.Data() != nullptr);
      REQUIRE_PRINT(bitmap.Size() > 0);
      REQUIRE_PRINT(!bitmap.Empty());

      auto queryKeys = aclco::test::roaring_bitmap_factory::GenerateQueryKeys<Key>(
          1024u, "mixed", std::vector<Key>{});
      auto results = aclco::test::roaring_bitmap_factory::ContainsKeys<Key>(bitmap, queryKeys, stream);
      REQUIRE_PRINT(results.size() == 1024u);
    }
  }
}
