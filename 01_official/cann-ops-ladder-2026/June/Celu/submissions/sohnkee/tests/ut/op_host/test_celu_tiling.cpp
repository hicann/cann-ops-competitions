#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "celu_tiling_data.h"

namespace CeluUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "Celu";

struct CeluTestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

static CeluTestParam testCases[] = {
    {"celu_0", {1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "0 0 0 ", {0}, 64, 262144, 4096},
};

class CeluTilingTest : public testing::TestWithParam<CeluTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "CeluTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "CeluTilingTest TearDown." << std::endl;
    }
};

struct CeluCompileInfo {} compileInfo;

static void TestOneParamCase(const CeluTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;

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

TEST_P(CeluTilingTest, tiling_test)
{
    const CeluTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    CeluTilingTests,
    CeluTilingTest,
    testing::ValuesIn(testCases));

}
