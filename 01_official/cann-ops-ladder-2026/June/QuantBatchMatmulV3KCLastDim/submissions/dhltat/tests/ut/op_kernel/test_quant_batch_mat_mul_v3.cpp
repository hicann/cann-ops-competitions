/*!
 * \file test_quant_batch_mat_mul_v3.cpp
 * \brief QuantBatchMatMulV3 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "quant_batch_mat_mul_v3_tiling.h"
#include "../../../op_kernel/quant_batch_mat_mul_v3.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

class QuantBatchMatMulV3KernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "QuantBatchMatMulV3KernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "QuantBatchMatMulV3KernelTest TearDown" << endl;
    }
};

TEST_F(QuantBatchMatMulV3KernelTest, test_kernel_run)
{
    constexpr size_t size = 8388608;
    constexpr size_t tilingDataSize = sizeof(QuantBatchMatMulV3TilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t x1ByteSize = size * sizeof(int8_t);
    constexpr size_t x2ByteSize = size * sizeof(int8_t);
    constexpr size_t scaleByteSize = size * sizeof(int8_t);
    constexpr size_t pertokenScaleByteSize = size * sizeof(int8_t);
    constexpr size_t outByteSize = size * sizeof(int8_t);
    std::vector<int8_t> x1Host(size, 1);
    std::vector<int8_t> x2Host(size, 1);
    std::vector<int8_t> scaleHost(size, 1);
    std::vector<int8_t> pertokenScaleHost(size, 1);
    std::vector<int8_t> outHost(size, 0.0f);
    
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x2ByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* pertokenScale = (uint8_t*)AscendC::GmAlloc(pertokenScaleByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    memcpy(x1, x1Host.data(), x1ByteSize);
    memcpy(x2, x2Host.data(), x2ByteSize);
    memcpy(scale, scaleHost.data(), scaleByteSize);
    memcpy(pertokenScale, pertokenScaleHost.data(), pertokenScaleByteSize);
    
    // 直接构造 tilingData（固定值，生成时确定）
    QuantBatchMatMulV3TilingData* tilingData = reinterpret_cast<QuantBatchMatMulV3TilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((quant_batch_mat_mul_v3<0>), numBlocks, x1, x2, scale, pertokenScale, out, workspace, tiling);
    
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(scale);
    AscendC::GmFree(pertokenScale);
    AscendC::GmFree(out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
