import os
import random
import numpy as np


def constraint_condition(case):
    """
    Huber Loss 测试用例约束条件

    Args:
        case: 测试用例字典

    Returns:
        case: 修改后的测试用例
    """
    case["expect_func"] = "{}:calc_expect_func".format(os.path.abspath("golden.py"))

    # 设置输入张量的取值范围
    for input_desc in case["input_desc"]:
        if input_desc["name"] == "input":
            input_desc["value_range"] = [-10, 10]
        elif input_desc["name"] == "target":
            input_desc["value_range"] = [-10, 10]

    # 获取输入 shape
    input_shape = case["input_desc"][0]["shape"]

    # 根据 reduction 模式设置输出 shape
    reduction_value = 1
    for attr_desc in case["attr_desc"]:
        if attr_desc["name"] == "reduction":
            attr_desc["value"] = random.choice([0, 1, 2])
            reduction_value = attr_desc["value"]
        elif attr_desc["name"] == "delta":
            # delta 必须 > 0，设置合理的取值范围
            attr_desc["value"] = random.choice([0.5, 1.0, 2.0, 5.0])

    # reduction != 0 时，输出为标量 [1]
    if reduction_value != 0:
        for output_desc in case["output_desc"]:
            output_desc["shape"] = [1]
    else:
        # reduction == 0 时，输出与输入 shape 相同
        for output_desc in case["output_desc"]:
            output_desc["shape"] = list(input_shape)

    return case