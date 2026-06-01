/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_roll_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "roll_tiling.h"
#include "../../../op_kernel/roll_tiling_data.h"
#include "../../../op_kernel/roll_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class RollTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "RollTiling SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "RollTiling TearDown" << endl;
    }
};

TEST_F(RollTiling, tiling_int32_axis0_success)
{
    RollCompileInfo compileInfo = {64};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{17, 64}, {17, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{17, 64}, {17, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0}))},
        &compileInfo);
    uint64_t expectTilingKey = 30000;
    string expectTilingData =
        "0 17 1 1 0 0 1 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 32736 2 17 64 0 0 0 0 0 0 64 1 0 "
        "0 0 0 0 0 3 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RollTiling, tiling_int32_multi_axis_success)
{
    RollCompileInfo compileInfo = {64};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{33, 54, 71}, {33, 54, 71}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{33, 54, 71}, {33, 54, 71}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 1, 4})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1, 2}))},
        &compileInfo);
    uint64_t expectTilingKey = 30001;
    string expectTilingData =
        "0 64 28 18 1 1 1 28 28 1 1 18 18 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 1977 1971 1 32736 3 33 54 71 0 "
        "0 0 0 0 3834 71 1 0 0 0 0 0 2 1 4 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RollTiling, tiling_small_tail_axis0_populates_simd_core_split)
{
    RollCompileInfo compileInfo = {64};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0}))},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingKey, 50001);
    ASSERT_EQ(tilingInfo.blockNum, 1U);

    auto* tilingData = reinterpret_cast<const RollTilingData*>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->usedCoreNum, 1);
    EXPECT_EQ(tilingData->perCoreElements, 12);
    EXPECT_EQ(tilingData->lastCoreElements, 12);
    EXPECT_GT(tilingData->ubElements, 0);
}

TEST_F(RollTiling, tiling_invalid_dim_range_failed)
{
    RollCompileInfo compileInfo = {64};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1, 7}))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
