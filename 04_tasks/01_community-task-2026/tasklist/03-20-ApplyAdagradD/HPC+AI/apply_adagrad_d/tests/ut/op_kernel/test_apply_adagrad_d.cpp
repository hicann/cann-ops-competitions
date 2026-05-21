/**
 * Copyright (c) 2026Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <cstdint>
#include <iostream>
#include <string>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#endif

#include "../../../op_kernel/apply_adagrad_d.cpp"
#include "../../../op_kernel/apply_adagrad_d_tiling_data.h"

using namespace std;

class ApplyAdagradDTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ApplyAdagradDTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "ApplyAdagradDTest TearDown" << endl;
    }
};

TEST_F(ApplyAdagradDTest, test_case_0)
{
    constexpr uint32_t dataNum = 32 * 4 * 4 * 4;
    size_t varByteSize   = dataNum * sizeof(float);
    size_t accumByteSize = dataNum * sizeof(float);
    size_t lrByteSize    = 1 * sizeof(float);
    size_t gradByteSize  = dataNum * sizeof(float);
    size_t varOutByteSize   = dataNum * sizeof(float);
    size_t accumOutByteSize = dataNum * sizeof(float);
    size_t tilingDataSize = sizeof(ApplyAdagradDTilingData);
    uint32_t blockDim = 1;

    uint8_t* var= reinterpret_cast<uint8_t*>(AscendC::GmAlloc(varByteSize));
    uint8_t* accum     = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(accumByteSize));
    uint8_t* lr        = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(lrByteSize));
    uint8_t* grad      = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gradByteSize));
    uint8_t* var_out   = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(varOutByteSize));
    uint8_t* accum_out = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(accumOutByteSize));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024 * 1024 * 16));
    uint8_t* tiling    = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));

    ASSERT_NE(var,nullptr);
    ASSERT_NE(accum,     nullptr);
    ASSERT_NE(lr,        nullptr);
    ASSERT_NE(grad,      nullptr);
    ASSERT_NE(var_out,   nullptr);
    ASSERT_NE(accum_out, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling,    nullptr);

    ApplyAdagradDTilingData* tilingData = reinterpret_cast<ApplyAdagradDTilingData*>(tiling);

    tilingData->smallCoreDataNum= 2048;
    tilingData->bigCoreDataNum    = 2112;
    tilingData->tileDataNum       = 4032;
    tilingData->smallTailDataNum  = 2048;
    tilingData->bigTailDataNum    = 2112;
    tilingData->finalSmallTileNum = 1;
    tilingData->finalBigTileNum   = 1;
    tilingData->tailBlockNum      = 0;

    auto KernelApplyAdagradD = [](GM_ADDR var, GM_ADDR accum, GM_ADDR lr, GM_ADDR grad,
                                  GM_ADDR var_out, GM_ADDR accum_out,
                                  GM_ADDR workspace, GM_ADDR tiling) {
        ::apply_adagrad_d<0>(var, accum, lr, grad, var_out, accum_out, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelApplyAdagradD, blockDim, var, accum, lr, grad,
                var_out, accum_out, workspace, reinterpret_cast<uint8_t*>(tilingData));

    AscendC::GmFree(var);
    AscendC::GmFree(accum);
    AscendC::GmFree(lr);
    AscendC::GmFree(grad);
    AscendC::GmFree(var_out);
    AscendC::GmFree(accum_out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
