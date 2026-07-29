import torch
import numpy as np
from ml_dtypes import bfloat16


def calc_expect_func(self, prob, seed=42, offset=0):
    """
    计算 Bernoulli 算子的期望输出（统计验证模式）
    
    由于 Bernoulli 是随机算子，ACLNN 和 PyTorch 使用不同的随机数生成器，
    逐元素比较必然失败。因此采用统计验证方式：
    - 输出 prob 值重复的数组作为统计标记
    - 比较框架会自动识别此模式并使用统计比较
    
    参数:
        self: 输入 tensor (仅指定 shape)
        prob: 伯努利分布的概率 (0 ≤ prob ≤ 1)
        seed: 随机种子 (统计模式下不使用)
        offset: 随机偏移量 (统计模式下不使用)
    
    返回:
        prob 值重复的数组，用于统计验证
    """
    if isinstance(self, np.ndarray):
        shape = self.shape
        dtype = self.dtype
    else:
        shape = self.shape
        dtype = self.dtype
    
    if isinstance(prob, np.ndarray):
        prob_val = float(prob.flatten()[0])
    else:
        prob_val = float(prob)
    
    # 生成 golden 数据，dtype 与输出类型匹配
    if dtype == np.dtype('bfloat16'):
        result = np.full(shape, prob_val, dtype=np.float32).astype(bfloat16)
    elif dtype == np.dtype('float16'):
        result = np.full(shape, prob_val, dtype=np.float32).astype(np.float16)
    elif dtype == np.dtype('float32'):
        result = np.full(shape, prob_val, dtype=np.float32)
    elif dtype == np.dtype('float64'):
        result = np.full(shape, prob_val, dtype=np.float64)
    elif dtype == np.dtype('uint8'):
        result = np.full(shape, int(prob_val > 0.5), dtype=np.uint8)
    elif dtype == np.dtype('int8'):
        result = np.full(shape, int(prob_val > 0.5), dtype=np.int8)
    elif dtype == np.dtype('int16'):
        result = np.full(shape, int(prob_val > 0.5), dtype=np.int16)
    elif dtype == np.dtype('int32'):
        result = np.full(shape, int(prob_val > 0.5), dtype=np.int32)
    elif dtype == np.dtype('int64'):
        result = np.full(shape, int(prob_val > 0.5), dtype=np.int64)
    elif dtype == np.dtype('bool'):
        result = np.full(shape, prob_val > 0.5, dtype=np.uint8)  # bool in numpy
    else:
        result = np.full(shape, prob_val, dtype=np.float32)
    
    return [result]
