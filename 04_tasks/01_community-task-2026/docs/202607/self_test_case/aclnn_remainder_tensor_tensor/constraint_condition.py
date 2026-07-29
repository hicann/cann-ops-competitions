import numpy as np
import math
import os
import random
from typing import Tuple, Union

def broadcast_shapes(shapes: list[Tuple[int, ...]]) -> Tuple[int, ...]:
    result = []
    for dims in zip(*[reversed(s) for s in shapes]):
        unique = set(dims)
        if len(unique) == 1:
            result.append(dims[0])
        elif 1 in unique:
            val = next(d for d in dims if d != 1)
            if any(d not in (1, val) for d in dims):
                raise ValueError(f"无法广播，维度组合{dims}")
            result.append(val)
        else:
            raise ValueError(f"维度无法广播: {dims}")
    return tuple(reversed(result))


def split_target_shape_to_broadcast(
    target_shape: Union[list[int], Tuple[int, ...]],
    seed: int = None
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    将目标输出shape随机拆分为两个可广播输入shape，broadcast(a,b) == target_shape
    约束：
    1. ndim < 2：直接返回两份原shape，不构造广播
    2. ndim >=2：最多仅2个维度启用广播，其余维度两者保持一致
    【修复】自动统一转为tuple，消除list/tuple比较失败问题
    """
    if seed is not None:
        random.seed(seed)

    # ==========关键修复1：强制统一转tuple============
    target_shape = tuple(target_shape)
    ndim = len(target_shape)

    if ndim < 2:
        return target_shape, target_shape

    a_dims = list(target_shape)
    b_dims = list(target_shape)

    num_broadcast_dim = random.choice([1, 2])
    broadcast_dim_indices = random.sample(range(ndim), k=num_broadcast_dim)

    for idx in broadcast_dim_indices:
        d = target_shape[idx]
        opt = random.choice([(d, 1), (1, d)])
        a_dims[idx], b_dims[idx] = opt

    shape_a = tuple(a_dims)
    shape_b = tuple(b_dims)

    out = broadcast_shapes([shape_a, shape_b])
    # ==========关键修复2：打印日志方便定位，不直接硬assert============
    if out != target_shape:
        raise RuntimeError(
            f"shape拆分非法!\n"
            f"target={target_shape}\n"
            f"shape_a={shape_a}, shape_b={shape_b}\n"
            f"broadcast_result={out}"
        )
    return shape_a, shape_b

def constraint_condition(case):
    case["expect_func"]="{}:calc_expect_func".format(os.path.abspath("golden.py"))
    for input_desc in case["input_desc"]:
        if input_desc["name"] == "x2":
            value_range = input_desc["value_range"]
            if random.choice([True, False]):
                if value_range[0] < 0:
                    value_range[1] = -1.001
                else:
                    value_range[0] = 1.001
            else:
                if value_range[1] > 1:
                    value_range[0] = 1.001
                else:
                    value_range[1] = -1.001

            input_desc["value_range"] = value_range
    shape = case["input_desc"][0]["shape"]

    # if random.choice([True, False]):
    # shape_a, shape_b = split_target_shape_to_broadcast(shape)
    # case["input_desc"][0]["shape"] = list(shape_a)
    # case["input_desc"][1]["shape"] = list(shape_b)
    # else:
    case["input_desc"][1]["shape"] = []

    return case
