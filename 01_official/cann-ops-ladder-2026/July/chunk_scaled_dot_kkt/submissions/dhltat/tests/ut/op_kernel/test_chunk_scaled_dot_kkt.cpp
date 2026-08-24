/*!
 * \file test_chunk_scaled_dot_kkt.cpp
 * \brief ChunkScaledDotKkt 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "chunk_scaled_dot_kkt_tiling.h"
#include "../../../op_kernel/chunk_scaled_dot_kkt.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

static uint16_t FloatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

class ChunkScaledDotKktKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ChunkScaledDotKktKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ChunkScaledDotKktKernelTest TearDown" << endl;
    }
};

TEST_F(ChunkScaledDotKktKernelTest, test_kernel_run)
{
    constexpr size_t size = 2564096;
    constexpr size_t tilingDataSize = sizeof(ChunkScaledDotKktTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t kByteSize = 2564096 * 2;
    constexpr size_t betaByteSize = 80128 * 2;
    constexpr size_t g_cumsumByteSize = 80128 * 4;
    constexpr size_t chunk_offsetsByteSize = 160 * 4;
    constexpr size_t AByteSize = 5128192 * 4;
    std::vector<float> kHost(2564096, 1);
    std::vector<float> betaHost(80128, 1);
    std::vector<float> g_cumsumHost(80128, 1);
    std::vector<int32_t> chunk_offsetsHost(160, 1);
    std::vector<float> AHost(5128192, 0);
    
    
    uint8_t* k = (uint8_t*)AscendC::GmAlloc(kByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* g_cumsum = (uint8_t*)AscendC::GmAlloc(g_cumsumByteSize);
    uint8_t* chunk_offsets = (uint8_t*)AscendC::GmAlloc(chunk_offsetsByteSize);
    uint8_t* A = (uint8_t*)AscendC::GmAlloc(AByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    for (size_t _i = 0; _i < 2564096; _i++) { uint16_t _b = FloatToBFloat16(kHost[_i]); memcpy(k + _i * 2, &_b, 2); }
    for (size_t _i = 0; _i < 80128; _i++) { uint16_t _b = FloatToBFloat16(betaHost[_i]); memcpy(beta + _i * 2, &_b, 2); }
    memcpy(g_cumsum, g_cumsumHost.data(), g_cumsumByteSize);
    memcpy(chunk_offsets, chunk_offsetsHost.data(), chunk_offsetsByteSize);
    
    // 直接构造 tilingData（固定值，生成时确定）
    ChunkScaledDotKktTilingData* tilingData = reinterpret_cast<ChunkScaledDotKktTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((chunk_scaled_dot_kkt<0>), numBlocks, k, beta, g_cumsum, chunk_offsets, A, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(AHost.data(), A, AByteSize);
    { std::ofstream _ofs("float32_output_chunk_scaled_dot_kkt_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(AHost.data()), AByteSize); }
    
    AscendC::GmFree(k);
    AscendC::GmFree(beta);
    AscendC::GmFree(g_cumsum);
    AscendC::GmFree(chunk_offsets);
    AscendC::GmFree(A);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
