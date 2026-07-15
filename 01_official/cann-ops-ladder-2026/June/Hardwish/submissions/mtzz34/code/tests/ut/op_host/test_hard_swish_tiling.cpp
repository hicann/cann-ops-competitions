#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "hard_swish_tiling_data.h"

namespace HardSwishUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "HardSwish";

struct HardSwishTestParam {
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

static HardSwishTestParam testCases[] = {
    {"hard_swish_0", {1}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "0 0 0 ", {0}, 64, 262144, 4096},
};

class HardSwishTilingTest : public testing::TestWithParam<HardSwishTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "HardSwishTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "HardSwishTilingTest TearDown." << std::endl;
    }
};

struct HardSwishCompileInfo {} compileInfo;

static void TestOneParamCase(const HardSwishTestParam &param)
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

TEST_P(HardSwishTilingTest, tiling_test)
{
    const HardSwishTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    HardSwishTilingTests,
    HardSwishTilingTest,
    testing::ValuesIn(testCases));

}
