#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

import numpy as np

def impl(x, shifts, dims):
    """
    纯 NumPy 实现 Roll 算子
    输入输出和 PyTorch 版完全一样，无任何依赖

    Roll 按照指定的维度和滚动量对张量进行循环滚动
    正数表示向右/向下滚动，负数表示向左/向上滚动

    Args:
        x: 输入张量
        shifts: 滚动量列表，每个元素对应一个维度的滚动量
        dims: 维度列表，指定要滚动的维度

    Returns:
        y: 滚动后的张量，与输入同形状、同类型
    """
    # 确保 shifts 和 dims 是列表
    if not isinstance(shifts, list):
        shifts = [shifts]
    if not isinstance(dims, list):
        dims = [dims]

    # 对每个维度进行滚动
    result = x.copy()
    for shift, dim in zip(shifts, dims):
        result = np.roll(result, shift, axis=dim)

    return result.astype(x.dtype)


if __name__ == "__main__":
    # 清理bin文件
    os.system("rm -rf *.bin")

    # 从 JSON 第一个 case 获取参数
    d_type = "float32"
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "int8": np.int8,
    }
    np_type = d_type_dict[d_type]

    # 生成输入数据
    input_x = np.random.uniform(-1000000.0, 1000000.0, (3)).astype(np_type)
    attr_shifts = [1]
    attr_dims = [0]

    # 计算 golden 数据
    golden = impl(input_x, attr_shifts, attr_dims)

    # 保存数据到文件
    input_x.astype(np_type).tofile(f"{d_type}_input_roll_x.bin")
    golden.astype(np_type).tofile(f"{d_type}_golden_roll.bin")

    print(f"生成完成: dtype={d_type}")
