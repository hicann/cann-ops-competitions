#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

import torch
import numpy
import torch.nn.functional as F
def impl(x, alpha):
    x = torch.from_numpy(x)
    x = x.detach().cpu()
    y = F.celu(x, alpha=alpha)
    return y.numpy()


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
    input_x = np.random.uniform(-5, 5, (1)).astype(np_type)

    
    # 计算 golden 数据
    golden = impl(input_x)
    
    # 保存数据到文件
    input_x.astype(np_type).tofile(f"{d_type}_input_celu_x.bin")
    golden.astype(np_type).tofile(f"{d_type}_golden_celu.bin")
    
    print(f"生成完成: dtype={d_type}")
