/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include <array>
#include <vector>

#include "../../../op_host/op_api/aclnn_gelu_backward_v2.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include <unistd.h>

using namespace op;
using namespace std;

class l2_gelu_backward_v2_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "gelu_backward_v2_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "gelu_backward_v2_test TearDown" << std::endl;
  }
};

TEST_F(l2_gelu_backward_v2_test, aclnnGeluBackwardV2_01_float_nd_none) {
  auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto selfDesc = TensorDesc({1, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  char* approximate = "none";

  auto outDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}
