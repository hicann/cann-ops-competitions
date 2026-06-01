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
 * \file test_roll.cpp
 * \brief Roll kernel UT.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/roll.cpp"

using namespace std;

class RollKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "roll kernel test SetUp" << endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./roll_data/");
    }

    static void TearDownTestCase()
    {
        cout << "roll kernel test TearDown" << endl;
    }

private:
    const static string rootPath;
    const static string dataPath;
};

const string RollKernelTest::rootPath = "../../../../experimental/";
const string RollKernelTest::dataPath = rootPath + "math/roll/tests/ut/op_kernel/roll_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(RollKernelTest, test_case_float32_dim0)
{
    constexpr uint32_t blockDim = 1;
    constexpr uint32_t dataCount = 12;
    constexpr uint32_t rows = 4;
    constexpr uint32_t cols = 3;
    constexpr uint32_t shift = 1;
    constexpr uint32_t ubElements = 1024;

    ASSERT_EQ(system("cd ./roll_data/ && python3 gen_data.py"), 0);

    size_t inputByteSize = dataCount * sizeof(float);
    uint8_t *x = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(CeilAlign(inputByteSize, 256)));
    string xFileName = "./roll_data/float32_input_t_roll.bin";
    ASSERT_TRUE(ReadFile(xFileName, inputByteSize, x, inputByteSize));

    size_t outputByteSize = dataCount * sizeof(float);
    uint8_t *y = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(CeilAlign(outputByteSize, 256)));
    size_t workspaceSize = 16 * 1024 * 1024;
    uint8_t *workspace = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize));
    uint8_t *tiling = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(sizeof(RollTilingData)));

    RollTilingData *tilingData = reinterpret_cast<RollTilingData *>(tiling);
    memset(tilingData, 0, sizeof(RollTilingData));
    tilingData->totalNum = dataCount;
    tilingData->dimNum = 2;
    tilingData->perCoreElements = dataCount;
    tilingData->lastCoreElements = dataCount;
    tilingData->usedCoreNum = blockDim;
    tilingData->ubElements = ubElements;
    tilingData->blockFactor = dataCount;
    tilingData->ubFactor = ubElements;
    tilingData->activeDimCount = 1;
    tilingData->activeDim = 0;
    tilingData->outerSize = 1;
    tilingData->dimSize = rows;
    tilingData->innerSize = cols;
    tilingData->activeShift = shift;
    tilingData->useSafeUbShuffle = 0;
    tilingData->shapes[0] = rows;
    tilingData->shapes[1] = cols;
    tilingData->strides[0] = cols;
    tilingData->strides[1] = 1;
    tilingData->shifts[0] = shift;
    tilingData->shifts[1] = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(roll, blockDim, x, y, workspace, reinterpret_cast<uint8_t *>(tilingData));

    string yFileName = "./roll_data/float32_output_t_roll.bin";
    ASSERT_TRUE(WriteFile(yFileName, y, outputByteSize));

    AscendC::GmFree(reinterpret_cast<void *>(x));
    AscendC::GmFree(reinterpret_cast<void *>(y));
    AscendC::GmFree(reinterpret_cast<void *>(workspace));
    AscendC::GmFree(reinterpret_cast<void *>(tiling));

    ASSERT_EQ(system("cd ./roll_data/ && python3 compare_data.py"), 0);
}
