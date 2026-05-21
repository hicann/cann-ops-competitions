/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * ...
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "aclnn_apply_adagrad_d.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

// update_slots=true 的 GetWorkspaceSize wrapper
static aclnnStatus aclnnApplyAdagradDTrueGetWorkspaceSize(
    const aclTensor* var, const aclTensor* accum,
    const aclTensor* lr,  const aclTensor* grad,
    aclTensor* varOut, aclTensor* accumOut,
    uint64_t* workspaceSize, aclOpExecutor** executor) {
  return aclnnApplyAdagradDGetWorkspaceSize(
      var, accum, lr, grad, true, varOut, accumOut, workspaceSize, executor);
}

// update_slots=true 的执行函数 wrapper（签名与框架 api_func 一致）
static aclnnStatus aclnnApplyAdagradDTrue(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
  return aclnnApplyAdagradD(workspace, workspaceSize, executor, stream);
}

// update_slots=false 的 GetWorkspaceSize wrapper
static aclnnStatus aclnnApplyAdagradDFalseGetWorkspaceSize(
    const aclTensor* var, const aclTensor* accum,
    const aclTensor* lr,  const aclTensor* grad,
    aclTensor* varOut, aclTensor* accumOut,
    uint64_t* workspaceSize, aclOpExecutor** executor) {
  return aclnnApplyAdagradDGetWorkspaceSize(
      var, accum, lr, grad, false, varOut, accumOut, workspaceSize, executor);
}

// update_slots=false 的执行函数 wrapper
static aclnnStatus aclnnApplyAdagradDFalse(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
  return aclnnApplyAdagradD(workspace, workspaceSize, executor, stream);
}

class apply_adagrad_d_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "apply_adagrad_d_test SetUp" << endl; }
  static void TearDownTestCase() { cout << "apply_adagrad_d_test TearDown" << endl; }
};

TEST_F(apply_adagrad_d_test, test_apply_adagrad_d_nullptr) {
  auto var      = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND);
  auto accum    = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND);
  auto lr       = TensorDesc({1},     ACL_FLOAT, ACL_FORMAT_ND);
  auto grad     = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND);
  auto varOut   = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND);
  auto accumOut = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspaceSize = 0;

  auto ut = OP_API_UT(aclnnApplyAdagradDTrue,
                      INPUT((aclTensor *)nullptr, accum, lr, grad),
                      OUTPUT(varOut, accumOut));
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);

  auto ut1 = OP_API_UT(aclnnApplyAdagradDTrue,
                       INPUT(var, (aclTensor *)nullptr, lr, grad),
                       OUTPUT(varOut, accumOut));
  EXPECT_EQ(ut1.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);

  auto ut2 = OP_API_UT(aclnnApplyAdagradDTrue,
                       INPUT(var, accum, (aclTensor *)nullptr, grad),
                       OUTPUT(varOut, accumOut));
  EXPECT_EQ(ut2.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);

  auto ut3 = OP_API_UT(aclnnApplyAdagradDTrue,
                       INPUT(var, accum, lr, (aclTensor *)nullptr),
                       OUTPUT(varOut, accumOut));
  EXPECT_EQ(ut3.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

