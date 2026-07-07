/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h" // NOLINT
#include "ut_op_common.h"

class GeluGradV2 : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "GeluGradV2 Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GeluGradV2 Proto Test TearDown" << std::endl;
    }
};

std::vector<int64_t> ToVectorForGeluGradV2(const gert::Shape& shape) {
  size_t shape_size = shape.GetDimNum();
  std::vector<int64_t> shape_vec(shape_size, 0);

  for (size_t i = 0; i < shape_size; i++) {
    shape_vec[i] = shape.GetDim(i);
  }
  return shape_vec;
}

TEST_F(GeluGradV2, GeluGradV2_infershape_none_fp16_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GeluGradV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GeluGradV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType output_ref = ge::DT_FLOAT16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 1)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}
