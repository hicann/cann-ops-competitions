import numpy as np
import random
import os
from typing import Tuple, Union


def constraint_condition(case):
    case["expect_func"] = "{}:calc_expect_func".format(os.path.abspath("golden.py"))
    
    # Bernoulli 算子约束
    # self 输入: shape 0-8 维，数据类型 FLOAT16/FLOAT/DOUBLE/UINT8/INT8/INT16/INT32/INT64/BOOL/BFLOAT16
    # prob: scalar, 0≤prob≤1, 数据类型 FLOAT16/FLOAT/DOUBLE/BFLOAT16
    # seed: int64
    # offset: int64, offset % 4 == 0
    # out: 输出 tensor, 数据类型与 self 一致，shape 与 self 一致
    
    # 设置 self 的 value_range (Bernoulli 输出为 0 或 1)
    for input_desc in case["input_desc"]:
        if input_desc["name"] == "self":
            # self 的值会被替换为 Bernoulli 分布的输出
            # 输入值不影响输出结果，仅用于指定 shape
            input_desc["value_range"] = [0.0, 1.0]
        elif input_desc["name"] == "prob":
            # prob 约束: 0 ≤ prob ≤ 1
            # Bernoulli 算子中 prob 是 scalar，value_range 用于生成 scalar 值
            input_desc["value_range"] = [0.0, 1.0]
            input_desc["shape"] = []
    
    # 设置输出的 shape 与 self 一致
    self_shape = case["input_desc"][0]["shape"]
    for output_desc in case["output_desc"]:
        if output_desc["name"] == "out":
            output_desc["shape"] = list(self_shape)
    
    # 添加属性: seed 和 offset
    # seed: 随机种子
    # offset: 必须满足 offset % 4 == 0
    seed = random.randint(0, 1000000)
    offset = random.choice([0, 4, 8, 16, 32, 64, 128, 256])
    
    # 设置 attr
    if "attr_desc" not in case:
        case["attr_desc"] = []
    
    # Bernoulli 的 attr 在 op.json 中定义，但 seed 和 offset 通常作为 op 的参数
    # 根据接口文档，seed 和 offset 是参数而非属性
    
    return case
