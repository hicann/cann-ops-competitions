#include <iostream>
#include <gtest/gtest.h>
#include "opdev/platform.h"
#include "aclnn_roll.h"
#include "test_utils.h"

using namespace std;
using namespace op;
using namespace op_api_test;

class AclnnRollTest : public testing::Test {
protected:
    void SetUp() override {
        cout << "AclnnRollTest SetUp" << endl;
    }

    void TearDown() override {
        cout << "AclnnRollTest TearDown" << endl;
    }
};

TEST_F(AclnnRollTest, FloatDtypeSuccess) {
    auto x = TestTensorFactory::CreateTensor({3}, DataType::DT_FLOAT);
    ASSERT_NE(x, nullptr);
    auto y = TestTensorFactory::CreateTensor({3}, DataType::DT_FLOAT);
    ASSERT_NE(y, nullptr);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnRollGetWorkspaceSize(x, {1}, {0}, y, &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);

    TestTensorFactory::DestroyTensor(x);
    TestTensorFactory::DestroyTensor(y);
}
