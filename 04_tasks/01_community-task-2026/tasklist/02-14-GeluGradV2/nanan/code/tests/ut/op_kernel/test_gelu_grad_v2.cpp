/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_gelu_grad_v2_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void gelu_grad_v2(
        GM_ADDR dy, GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling);

class gelu_grad_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "gelu_grad_v2 SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "gelu_grad_v2_test TearDown\n" << endl;
    }
};

TEST_F(gelu_grad_v2_test, test_case_mode_1_fp32_001)
{
    size_t inputdyByteSize = 1024 * sizeof(float);
    size_t inputxByteSize = 1024 * sizeof(float);
    size_t outputzByteSize = 1024 * sizeof(float);
    size_t tilingDataSize = sizeof(GeluGradV2TilingDataTest);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(inputdyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputxByteSize);

    uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputzByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    GeluGradV2TilingData* tilingDatafromBin = reinterpret_cast<GeluGradV2TilingData*>(tiling);

    tilingDatafromBin->smallCoreDataNum = 1024;
    tilingDatafromBin->bigCoreDataNum = 1152;
    tilingDatafromBin->finalBigTileNum = 2;
    tilingDatafromBin->finalSmallTileNum = 1;
    tilingDatafromBin->tileDataNum = 1024;
    tilingDatafromBin->smallTailDataNum = 1024;
    tilingDatafromBin->bigTailDataNum = 128;
    tilingDatafromBin->tailBlockNum = 0;
    tilingDatafromBin->usedDb = 0;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(gelu_grad_v2, blockDim, dy, x, z, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

