/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <string>
#include <cstring>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif
#include "../../../op_kernel/relu_grad_v3.cpp"
#include "../../../op_kernel/relu_grad_v3_tiling_data.h"

using namespace std;

class relu_grad_v3_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "relu_grad_v3_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "relu_grad_v3_test TearDown\n" << endl;
    }
};

TEST_F(relu_grad_v3_test, test_case_float32)
{
    size_t gradByteSize = 8 * sizeof(float);
    size_t maskByteSize = 8 * sizeof(uint8_t);
    size_t outputByteSize = 8 * sizeof(float);
    size_t tilingDataSize = sizeof(ReluGradV3TilingData);
    uint32_t blockDim = 1;

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* backprops = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    float* gradData = reinterpret_cast<float*>(gradients);
    uint8_t* maskData = reinterpret_cast<uint8_t*>(mask);
    float* outData = reinterpret_cast<float*>(backprops);

    for (size_t i = 0; i < 8; ++i) {
        gradData[i] = static_cast<float>(i + 1);
        maskData[i] = (i % 2 == 0) ? 1 : 0;
    }

    float expected[8] = {1.0f, 0.0f, 3.0f, 0.0f, 5.0f, 0.0f, 7.0f, 0.0f};

    memset(backprops, 0, outputByteSize);

    ReluGradV3TilingData* tilingData = reinterpret_cast<ReluGradV3TilingData*>(tiling);
    tilingData->smallCoreDataNum = 8;
    tilingData->bigCoreDataNum = 8;
    tilingData->tileDataNum = 8;
    tilingData->smallTailDataNum = 8;
    tilingData->bigTailDataNum = 8;
    tilingData->finalSmallTileNum = 1;
    tilingData->finalBigTileNum = 1;
    tilingData->tailBlockNum = 0;

    auto KernelReluGradV3 = [](GM_ADDR gradients, GM_ADDR mask, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling) {
        ::relu_grad_v3<1>(gradients, mask, backprops, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(1); // schMode=1: float32
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelReluGradV3, blockDim, gradients, mask, backprops, workspace, (uint8_t *)(tilingData));

    int passCount = 0;
    for (size_t i = 0; i < 8; ++i) {
        float diff = fabsf(outData[i] - expected[i]);
        if (diff < 0.001f) {
            passCount++;
        } else {
            cout << "  Mismatch at index " << i << ": expected=" << expected[i]
                 << ", got=" << outData[i] << endl;
        }
    }

    AscendC::GmFree(gradients);
    AscendC::GmFree(mask);
    AscendC::GmFree(backprops);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);

    EXPECT_EQ(passCount, 8);
}

TEST_F(relu_grad_v3_test, test_case_float16)
{
    size_t gradByteSize = 16 * sizeof(half);
    size_t maskByteSize = 16 * sizeof(uint8_t);
    size_t outputByteSize = 16 * sizeof(half);
    size_t tilingDataSize = sizeof(ReluGradV3TilingData);
    uint32_t blockDim = 1;

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* backprops = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    half* gradData = reinterpret_cast<half*>(gradients);
    uint8_t* maskData = reinterpret_cast<uint8_t*>(mask);
    half* outData = reinterpret_cast<half*>(backprops);

    float gradVals[16] = {
        1.5f, -2.5f, 3.5f, -4.5f,
        5.5f, -6.5f, 7.5f, -8.5f,
        9.5f, -10.5f, 11.5f, -12.5f,
        13.5f, -14.5f, -0.0f, -16.5f
    };
    for (size_t i = 0; i < 16; ++i) {
        gradData[i] = static_cast<half>(gradVals[i]);
        maskData[i] = (i % 2 == 0) ? 1 : 0;
    }

    half expected[16];
    for (size_t i = 0; i < 16; ++i) {
        expected[i] = (maskData[i] != 0) ? gradData[i] : static_cast<half>(0);
    }

    memset(backprops, 0, outputByteSize);

    ReluGradV3TilingData* tilingData = reinterpret_cast<ReluGradV3TilingData*>(tiling);
    tilingData->smallCoreDataNum = 16;
    tilingData->bigCoreDataNum = 16;
    tilingData->tileDataNum = 16;
    tilingData->smallTailDataNum = 16;
    tilingData->bigTailDataNum = 16;
    tilingData->finalSmallTileNum = 1;
    tilingData->finalBigTileNum = 1;
    tilingData->tailBlockNum = 0;

    auto KernelReluGradV3 = [](GM_ADDR gradients, GM_ADDR mask, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling) {
        ::relu_grad_v3<0>(gradients, mask, backprops, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelReluGradV3, blockDim, gradients, mask, backprops, workspace, (uint8_t *)(tilingData));

    // 浮点容差比较：+0 == -0 在 IEEE 754 下为 true，语义正确
    int passCount = 0;
    for (size_t i = 0; i < 16; ++i) {
        float diff = fabsf(static_cast<float>(outData[i]) - static_cast<float>(expected[i]));
        if (diff < 0.01f) {
            passCount++;
        } else {
            cout << "  Mismatch at index " << i << ": expected=" << static_cast<float>(expected[i])
                 << ", got=" << static_cast<float>(outData[i]) << endl;
        }
    }

    AscendC::GmFree(gradients);
    AscendC::GmFree(mask);
    AscendC::GmFree(backprops);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);

    EXPECT_EQ(passCount, 16);
}

TEST_F(relu_grad_v3_test, test_case_int32)
{
    size_t gradByteSize = 8 * sizeof(int32_t);
    size_t maskByteSize = 8 * sizeof(uint8_t);
    size_t outputByteSize = 8 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(ReluGradV3TilingData);
    uint32_t blockDim = 1;

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* backprops = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    int32_t* gradData = reinterpret_cast<int32_t*>(gradients);
    uint8_t* maskData = reinterpret_cast<uint8_t*>(mask);
    int32_t* outData = reinterpret_cast<int32_t*>(backprops);

    gradData[0] = 10; gradData[1] = 20; gradData[2] = 30; gradData[3] = 40;
    gradData[4] = 50; gradData[5] = 60; gradData[6] = 70; gradData[7] = 80;
    maskData[0] = 1; maskData[1] = 0; maskData[2] = 1; maskData[3] = 0;
    maskData[4] = 1; maskData[5] = 0; maskData[6] = 1; maskData[7] = 0;

    int32_t expected[8];
    for (size_t i = 0; i < 8; ++i) {
        expected[i] = (maskData[i] != 0) ? gradData[i] : 0;
    }

    memset(backprops, 0, outputByteSize);

    ReluGradV3TilingData* tilingData = reinterpret_cast<ReluGradV3TilingData*>(tiling);
    tilingData->smallCoreDataNum = 8;
    tilingData->bigCoreDataNum = 8;
    tilingData->tileDataNum = 8;
    tilingData->smallTailDataNum = 8;
    tilingData->bigTailDataNum = 8;
    tilingData->finalSmallTileNum = 1;
    tilingData->finalBigTileNum = 1;
    tilingData->tailBlockNum = 0;

    auto KernelReluGradV3 = [](GM_ADDR gradients, GM_ADDR mask, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling) {
        ::relu_grad_v3<2>(gradients, mask, backprops, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelReluGradV3, blockDim, gradients, mask, backprops, workspace, (uint8_t *)(tilingData));

    int passCount = 0;
    for (size_t i = 0; i < 8; ++i) {
        if (outData[i] == expected[i]) {
            passCount++;
        } else {
            cout << "  Mismatch at index " << i << ": expected=" << expected[i]
                 << ", got=" << outData[i] << endl;
        }
    }

    AscendC::GmFree(gradients);
    AscendC::GmFree(mask);
    AscendC::GmFree(backprops);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);

    EXPECT_EQ(passCount, 8);
}

TEST_F(relu_grad_v3_test, test_case_int8)
{
    size_t gradByteSize = 8 * sizeof(int8_t);
    size_t maskByteSize = 8 * sizeof(uint8_t);
    size_t outputByteSize = 8 * sizeof(int8_t);
    size_t tilingDataSize = sizeof(ReluGradV3TilingData);
    uint32_t blockDim = 1;

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* backprops = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    int8_t* gradData = reinterpret_cast<int8_t*>(gradients);
    uint8_t* maskData = reinterpret_cast<uint8_t*>(mask);
    int8_t* outData = reinterpret_cast<int8_t*>(backprops);

    gradData[0] = 10; gradData[1] = -20; gradData[2] = 30; gradData[3] = -40;
    gradData[4] = 50; gradData[5] = -60; gradData[6] = 70; gradData[7] = -80;
    maskData[0] = 1; maskData[1] = 0; maskData[2] = 1; maskData[3] = 0;
    maskData[4] = 1; maskData[5] = 0; maskData[6] = 1; maskData[7] = 0;

    int8_t expected[8];
    for (size_t i = 0; i < 8; ++i) {
        expected[i] = (maskData[i] != 0) ? gradData[i] : 0;
    }

    memset(backprops, 0, outputByteSize);

    ReluGradV3TilingData* tilingData = reinterpret_cast<ReluGradV3TilingData*>(tiling);
    tilingData->smallCoreDataNum = 8;
    tilingData->bigCoreDataNum = 8;
    tilingData->tileDataNum = 8;
    tilingData->smallTailDataNum = 8;
    tilingData->bigTailDataNum = 8;
    tilingData->finalSmallTileNum = 1;
    tilingData->finalBigTileNum = 1;
    tilingData->tailBlockNum = 0;

    auto KernelReluGradV3 = [](GM_ADDR gradients, GM_ADDR mask, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling) {
        ::relu_grad_v3<3>(gradients, mask, backprops, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(3);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelReluGradV3, blockDim, gradients, mask, backprops, workspace, (uint8_t *)(tilingData));

    int passCount = 0;
    for (size_t i = 0; i < 8; ++i) {
        if (outData[i] == expected[i]) {
            passCount++;
        } else {
            cout << "  Mismatch at index " << i << ": expected=" << static_cast<int>(expected[i])
                 << ", got=" << static_cast<int>(outData[i]) << endl;
        }
    }

    AscendC::GmFree(gradients);
    AscendC::GmFree(mask);
    AscendC::GmFree(backprops);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);

    EXPECT_EQ(passCount, 8);
}

TEST_F(relu_grad_v3_test, test_case_uint8)
{
    size_t gradByteSize = 8 * sizeof(uint8_t);
    size_t maskByteSize = 8 * sizeof(uint8_t);
    size_t outputByteSize = 8 * sizeof(uint8_t);
    size_t tilingDataSize = sizeof(ReluGradV3TilingData);
    uint32_t blockDim = 1;

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* backprops = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    uint8_t* gradData = reinterpret_cast<uint8_t*>(gradients);
    uint8_t* maskData = reinterpret_cast<uint8_t*>(mask);
    uint8_t* outData = reinterpret_cast<uint8_t*>(backprops);

    gradData[0] = 10; gradData[1] = 20; gradData[2] = 30; gradData[3] = 40;
    gradData[4] = 50; gradData[5] = 60; gradData[6] = 70; gradData[7] = 80;
    maskData[0] = 1; maskData[1] = 0; maskData[2] = 1; maskData[3] = 0;
    maskData[4] = 1; maskData[5] = 0; maskData[6] = 1; maskData[7] = 0;

    uint8_t expected[8];
    for (size_t i = 0; i < 8; ++i) {
        expected[i] = (maskData[i] != 0) ? gradData[i] : 0;
    }

    memset(backprops, 0, outputByteSize);

    ReluGradV3TilingData* tilingData = reinterpret_cast<ReluGradV3TilingData*>(tiling);
    tilingData->smallCoreDataNum = 8;
    tilingData->bigCoreDataNum = 8;
    tilingData->tileDataNum = 8;
    tilingData->smallTailDataNum = 8;
    tilingData->bigTailDataNum = 8;
    tilingData->finalSmallTileNum = 1;
    tilingData->finalBigTileNum = 1;
    tilingData->tailBlockNum = 0;

    auto KernelReluGradV3 = [](GM_ADDR gradients, GM_ADDR mask, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling) {
        ::relu_grad_v3<4>(gradients, mask, backprops, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(4);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelReluGradV3, blockDim, gradients, mask, backprops, workspace, (uint8_t *)(tilingData));

    int passCount = 0;
    for (size_t i = 0; i < 8; ++i) {
        if (outData[i] == expected[i]) {
            passCount++;
        } else {
            cout << "  Mismatch at index " << i << ": expected=" << static_cast<int>(expected[i])
                 << ", got=" << static_cast<int>(outData[i]) << endl;
        }
    }

    AscendC::GmFree(gradients);
    AscendC::GmFree(mask);
    AscendC::GmFree(backprops);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);

    EXPECT_EQ(passCount, 8);
}

TEST_F(relu_grad_v3_test, test_case_bfloat16)
{
    size_t gradByteSize = 16 * sizeof(bfloat16_t);
    size_t maskByteSize = 16 * sizeof(uint8_t);
    size_t outputByteSize = 16 * sizeof(bfloat16_t);
    size_t tilingDataSize = sizeof(ReluGradV3TilingData);
    uint32_t blockDim = 1;

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* backprops = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    bfloat16_t* gradData = reinterpret_cast<bfloat16_t*>(gradients);
    uint8_t* maskData = reinterpret_cast<uint8_t*>(mask);
    bfloat16_t* outData = reinterpret_cast<bfloat16_t*>(backprops);

    float gradVals[16] = {
        1.5f, -2.5f, 3.5f, -4.5f,
        5.5f, -6.5f, 7.5f, -8.5f,
        9.5f, -10.5f, 11.5f, -12.5f,
        13.5f, -14.5f, -0.0f, -16.5f
    };
    for (size_t i = 0; i < 16; ++i) {
        gradData[i] = static_cast<bfloat16_t>(gradVals[i]);
        maskData[i] = (i % 2 == 0) ? 1 : 0;
    }

    // 期望输出: mask=1 时 = grad, mask=0 时 = +0 (bit pattern 0x0000)
    // 与 TBE VSEL 实现对齐：条件选择输出 +0
    bfloat16_t expected[16];
    for (size_t i = 0; i < 16; ++i) {
        expected[i] = (maskData[i] != 0) ? gradData[i] : static_cast<bfloat16_t>(0);
    }

    memset(backprops, 0, outputByteSize);

    ReluGradV3TilingData* tilingData = reinterpret_cast<ReluGradV3TilingData*>(tiling);
    tilingData->smallCoreDataNum = 16;
    tilingData->bigCoreDataNum = 16;
    tilingData->tileDataNum = 16;
    tilingData->smallTailDataNum = 16;
    tilingData->bigTailDataNum = 16;
    tilingData->finalSmallTileNum = 1;
    tilingData->finalBigTileNum = 1;
    tilingData->tailBlockNum = 0;

    auto KernelReluGradV3 = [](GM_ADDR gradients, GM_ADDR mask, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling) {
        ::relu_grad_v3<5>(gradients, mask, backprops, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(5);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelReluGradV3, blockDim, gradients, mask, backprops, workspace, (uint8_t *)(tilingData));

    // 浮点容差比较：+0 == -0 在 IEEE 754 下为 true，语义正确
    int passCount = 0;
    for (size_t i = 0; i < 16; ++i) {
        float diff = fabsf(static_cast<float>(outData[i]) - static_cast<float>(expected[i]));
        if (diff < 0.01f) {
            passCount++;
        } else {
            cout << "  Mismatch at index " << i << ": expected=" << static_cast<float>(expected[i])
                 << ", got=" << static_cast<float>(outData[i]) << endl;
        }
    }

    AscendC::GmFree(gradients);
    AscendC::GmFree(mask);
    AscendC::GmFree(backprops);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);

    EXPECT_EQ(passCount, 16);
}

// ========== 大 tile 测试（触发 Compares+Select 路径） ==========

TEST_F(relu_grad_v3_test, test_case_float32_large_tile)
{
    constexpr size_t N = 512;
    constexpr size_t tileDataNum = 256;  // >= CMP_ALIGN_ELEMS(128)
    size_t gradByteSize = N * sizeof(float);
    size_t maskByteSize = N * sizeof(uint8_t);
    size_t outputByteSize = N * sizeof(float);
    size_t tilingDataSize = sizeof(ReluGradV3TilingData);
    uint32_t blockDim = 1;

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* backprops = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    float* gradData = reinterpret_cast<float*>(gradients);
    uint8_t* maskData = reinterpret_cast<uint8_t*>(mask);
    float* outData = reinterpret_cast<float*>(backprops);

    float expected[N];
    for (size_t i = 0; i < N; ++i) {
        gradData[i] = (i % 3 == 0) ? -1.5f * (i + 1) : 2.0f * (i + 1);
        maskData[i] = (i % 3 == 0) ? 0 : 1;
        expected[i] = (maskData[i] != 0) ? gradData[i] : 0.0f;
    }

    memset(backprops, 0, outputByteSize);

    ReluGradV3TilingData* tilingData = reinterpret_cast<ReluGradV3TilingData*>(tiling);
    tilingData->smallCoreDataNum = N;
    tilingData->bigCoreDataNum = N;
    tilingData->tileDataNum = tileDataNum;
    tilingData->smallTailDataNum = N - ((N / tileDataNum - 1) * tileDataNum);
    tilingData->bigTailDataNum = tilingData->smallTailDataNum;
    tilingData->finalSmallTileNum = (N + tileDataNum - 1) / tileDataNum;
    tilingData->finalBigTileNum = tilingData->finalSmallTileNum;
    tilingData->tailBlockNum = 0;

    auto KernelReluGradV3 = [](GM_ADDR gradients, GM_ADDR mask, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling) {
        ::relu_grad_v3<1>(gradients, mask, backprops, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelReluGradV3, blockDim, gradients, mask, backprops, workspace, (uint8_t *)(tilingData));

    int passCount = 0;
    for (size_t i = 0; i < N; ++i) {
        float diff = fabsf(outData[i] - expected[i]);
        if (diff < 0.001f) passCount++;
    }

    AscendC::GmFree(gradients);
    AscendC::GmFree(mask);
    AscendC::GmFree(backprops);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    EXPECT_EQ(passCount, N) << "float32 large tile (Compares+Select) failed";
}

TEST_F(relu_grad_v3_test, test_case_float16_large_tile)
{
    constexpr size_t N = 512;
    constexpr size_t tileDataNum = 256;  // >= CMP_ALIGN_ELEMS(128)
    size_t gradByteSize = N * sizeof(half);
    size_t maskByteSize = N * sizeof(uint8_t);
    size_t outputByteSize = N * sizeof(half);
    size_t tilingDataSize = sizeof(ReluGradV3TilingData);
    uint32_t blockDim = 1;

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* backprops = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    half* gradData = reinterpret_cast<half*>(gradients);
    uint8_t* maskData = reinterpret_cast<uint8_t*>(mask);
    half* outData = reinterpret_cast<half*>(backprops);

    half expected[N];
    for (size_t i = 0; i < N; ++i) {
        gradData[i] = static_cast<half>((i % 3 == 0) ? -1.5f * (i + 1) : 2.0f * (i + 1));
        maskData[i] = (i % 3 == 0) ? 0 : 1;
        expected[i] = (maskData[i] != 0) ? gradData[i] : static_cast<half>(0);
    }

    memset(backprops, 0, outputByteSize);

    ReluGradV3TilingData* tilingData = reinterpret_cast<ReluGradV3TilingData*>(tiling);
    tilingData->smallCoreDataNum = N;
    tilingData->bigCoreDataNum = N;
    tilingData->tileDataNum = tileDataNum;
    tilingData->smallTailDataNum = N - ((N / tileDataNum - 1) * tileDataNum);
    tilingData->bigTailDataNum = tilingData->smallTailDataNum;
    tilingData->finalSmallTileNum = (N + tileDataNum - 1) / tileDataNum;
    tilingData->finalBigTileNum = tilingData->finalSmallTileNum;
    tilingData->tailBlockNum = 0;

    auto KernelReluGradV3 = [](GM_ADDR gradients, GM_ADDR mask, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling) {
        ::relu_grad_v3<0>(gradients, mask, backprops, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelReluGradV3, blockDim, gradients, mask, backprops, workspace, (uint8_t *)(tilingData));

    int passCount = 0;
    for (size_t i = 0; i < N; ++i) {
        float diff = fabsf(static_cast<float>(outData[i]) - static_cast<float>(expected[i]));
        if (diff < 0.01f) passCount++;
    }

    AscendC::GmFree(gradients);
    AscendC::GmFree(mask);
    AscendC::GmFree(backprops);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    EXPECT_EQ(passCount, N) << "float16 large tile (Compares+Select) failed";
}

TEST_F(relu_grad_v3_test, test_case_bfloat16_large_tile)
{
    constexpr size_t N = 512;
    constexpr size_t tileDataNum = 256;  // >= CMP_ALIGN_ELEMS(128), 触发 Select
    size_t gradByteSize = N * sizeof(bfloat16_t);
    size_t maskByteSize = N * sizeof(uint8_t);
    size_t outputByteSize = N * sizeof(bfloat16_t);
    size_t tilingDataSize = sizeof(ReluGradV3TilingData);
    uint32_t blockDim = 1;

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* backprops = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    bfloat16_t* gradData = reinterpret_cast<bfloat16_t*>(gradients);
    uint8_t* maskData = reinterpret_cast<uint8_t*>(mask);
    bfloat16_t* outData = reinterpret_cast<bfloat16_t*>(backprops);

    bfloat16_t expected[N];
    for (size_t i = 0; i < N; ++i) {
        gradData[i] = static_cast<bfloat16_t>((i % 3 == 0) ? -1.5f * (i + 1) : 2.0f * (i + 1));
        maskData[i] = (i % 3 == 0) ? 0 : 1;
        expected[i] = (maskData[i] != 0) ? gradData[i] : static_cast<bfloat16_t>(0);
    }

    memset(backprops, 0, outputByteSize);

    ReluGradV3TilingData* tilingData = reinterpret_cast<ReluGradV3TilingData*>(tiling);
    tilingData->smallCoreDataNum = N;
    tilingData->bigCoreDataNum = N;
    tilingData->tileDataNum = tileDataNum;
    tilingData->smallTailDataNum = N - ((N / tileDataNum - 1) * tileDataNum);
    tilingData->bigTailDataNum = tilingData->smallTailDataNum;
    tilingData->finalSmallTileNum = (N + tileDataNum - 1) / tileDataNum;
    tilingData->finalBigTileNum = tilingData->finalSmallTileNum;
    tilingData->tailBlockNum = 0;

    auto KernelReluGradV3 = [](GM_ADDR gradients, GM_ADDR mask, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling) {
        ::relu_grad_v3<5>(gradients, mask, backprops, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(5);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelReluGradV3, blockDim, gradients, mask, backprops, workspace, (uint8_t *)(tilingData));

    int passCount = 0;
    for (size_t i = 0; i < N; ++i) {
        float diff = fabsf(static_cast<float>(outData[i]) - static_cast<float>(expected[i]));
        if (diff < 0.01f) passCount++;
    }

    AscendC::GmFree(gradients);
    AscendC::GmFree(mask);
    AscendC::GmFree(backprops);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    EXPECT_EQ(passCount, N) << "bfloat16 large tile (Compares+Select) failed";
}