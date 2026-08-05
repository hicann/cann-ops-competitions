import torch
import torch.nn.functional as F
import numpy as np
from ml_dtypes import bfloat16


def calc_expect_func(input, target, reduction=1, delta=1.0):
    """
    Huber Loss 标杆计算函数

    Args:
        input: 预测值张量
        target: 目标值张量
        reduction: 归约模式，0=none, 1=mean, 2=sum
        delta: Huber loss 阈值参数

    Returns:
        [result]: 计算结果
    """
    reduction_map = {
        0: "none",
        1: "mean",
        2: "sum"
    }
    reduction_str = reduction_map.get(reduction, "mean")

    # 处理 bfloat16 类型
    isBf16 = input.dtype == bfloat16
    if isBf16:
        input_tensor = torch.from_numpy(input.astype(np.float32)).bfloat16()
        target_tensor = torch.from_numpy(target.astype(np.float32)).bfloat16()
    else:
        input_tensor = torch.from_numpy(input)
        target_tensor = torch.from_numpy(target)

    ori_type = input_tensor.dtype
    if ori_type == torch.half or ori_type == torch.bfloat16:
        input_tensor = input_tensor.float()
        target_tensor = target_tensor.float()

    result = F.huber_loss(input_tensor, target_tensor, reduction=reduction_str, delta=delta)
    result = result.to(ori_type)

    # 转回 numpy
    if isBf16:
        res = result.float().numpy().astype(bfloat16)
    else:
        res = result.numpy()

    # reduction != 0 时，输出为标量
    if reduction != 0:
        res = np.array([res], dtype=res.dtype)

    return [res]