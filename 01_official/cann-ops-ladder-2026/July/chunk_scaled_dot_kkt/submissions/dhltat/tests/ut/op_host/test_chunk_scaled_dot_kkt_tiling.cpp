#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "chunk_scaled_dot_kkt_tiling_data.h"

namespace ChunkScaledDotKktUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "ChunkScaledDotKkt";

struct ChunkScaledDotKktTestParam {
    std::string caseName;
    std::initializer_list<int64_t> kShape;
    ge::DataType kDtype;
    ge::Format kFormat;
    std::initializer_list<int64_t> betaShape;
    ge::DataType betaDtype;
    ge::Format betaFormat;
    std::initializer_list<int64_t> g_cumsumShape;
    ge::DataType g_cumsumDtype;
    ge::Format g_cumsumFormat;
    std::initializer_list<int64_t> chunk_offsetsShape;
    ge::DataType chunk_offsetsDtype;
    ge::Format chunk_offsetsFormat;
    std::initializer_list<int64_t> AShape;
    ge::DataType ADtype;
    ge::Format AFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

static ChunkScaledDotKktTestParam testCases[] = {
    {"chunk_scaled_dot_kkt_0", {1, 10016, 2, 128}, ge::DT_BF16, ge::FORMAT_ND, {1, 10016, 8}, ge::DT_BF16, ge::FORMAT_ND, {1, 10016, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {160}, ge::DT_INT32, ge::FORMAT_ND, {1, 10016, 8, 64}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class ChunkScaledDotKktTilingTest : public testing::TestWithParam<ChunkScaledDotKktTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "ChunkScaledDotKktTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "ChunkScaledDotKktTilingTest TearDown." << std::endl;
    }
};

struct ChunkScaledDotKktCompileInfo {} compileInfo;

static void TestOneParamCase(const ChunkScaledDotKktTestParam &param)
{
    gert::StorageShape kShape = {param.kShape, param.kShape};
    gert::StorageShape betaShape = {param.betaShape, param.betaShape};
    gert::StorageShape g_cumsumShape = {param.g_cumsumShape, param.g_cumsumShape};
    gert::StorageShape chunk_offsetsShape = {param.chunk_offsetsShape, param.chunk_offsetsShape};
    gert::StorageShape AShape = {param.AShape, param.AShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{kShape, param.kDtype, param.kFormat},
        {betaShape, param.betaDtype, param.betaFormat},
        {g_cumsumShape, param.g_cumsumDtype, param.g_cumsumFormat},
        {chunk_offsetsShape, param.chunk_offsetsDtype, param.chunk_offsetsFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{AShape, param.ADtype, param.AFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("chunk_size", Ops::Math::AnyValue::CreateFrom<int64_t>(64)));
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        param.maxAIVNum,
        param.ubSize,
        param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey,
                    param.expectTilingData, param.expectWorkspaces);
}

TEST_P(ChunkScaledDotKktTilingTest, tiling_test)
{
    const ChunkScaledDotKktTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    ChunkScaledDotKktTilingTests,
    ChunkScaledDotKktTilingTest,
    testing::ValuesIn(testCases));

}
