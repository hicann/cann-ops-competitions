#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "roll_tiling_data.h"

namespace RollUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "Roll";

struct RollTestParam {
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

static RollTestParam testCases[] = {
    {"roll_0", {3}, ge::DT_FLOAT, ge::FORMAT_ND, {3}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "0 0 0 ", {0}, 64, 262144, 4096},
};

class RollTilingTest : public testing::TestWithParam<RollTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "RollTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "RollTilingTest TearDown." << std::endl;
    }
};

struct RollCompileInfo {} compileInfo;

static void TestOneParamCase(const RollTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})));
    attrs_.push_back(gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})));
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

TEST_P(RollTilingTest, tiling_test)
{
    const RollTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    RollTilingTests,
    RollTilingTest,
    testing::ValuesIn(testCases));

}
