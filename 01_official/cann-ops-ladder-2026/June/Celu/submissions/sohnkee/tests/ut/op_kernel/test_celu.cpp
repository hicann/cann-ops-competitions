/*!
 * \file test_celu.cpp
 * \brief Celu 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "celu_tiling.h"
#include "../../../op_kernel/celu.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

class CeluKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "CeluKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "CeluKernelTest TearDown" << endl;
    }
};

TEST_F(CeluKernelTest, test_kernel_run)
{
    constexpr size_t size = 1;
    constexpr size_t tilingDataSize = sizeof(CeluTilingData);
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
    CeluTilingData* tilingData = reinterpret_cast<CeluTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((celu<0>), numBlocks, x, y, workspace, tiling);
    
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
