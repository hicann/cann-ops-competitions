/*!
 * \file test_selu.cpp
 * \brief Selu 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "selu_tiling.h"
#include "../../../op_kernel/selu.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

class SeluKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "SeluKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "SeluKernelTest TearDown" << endl;
    }
};

TEST_F(SeluKernelTest, test_kernel_run)
{
    constexpr size_t size = 1;
    constexpr size_t tilingDataSize = sizeof(SeluTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = size * sizeof(float);
    constexpr size_t yByteSize = size * sizeof(float);
    std::vector<float> xHost(size, 1);
    std::vector<float> yHost(size, 0.0f);
    
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    memcpy(x, xHost.data(), xByteSize);
    
    // 直接构造 tilingData（固定值，生成时确定）
    SeluTilingData* tilingData = reinterpret_cast<SeluTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((selu<0>), numBlocks, x, y, workspace, tiling);
    
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
