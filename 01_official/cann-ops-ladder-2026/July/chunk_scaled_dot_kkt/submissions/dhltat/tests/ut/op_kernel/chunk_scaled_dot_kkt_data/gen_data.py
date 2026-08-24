#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16



if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)
    
    # 从 JSON 第一个 case 获取参数
    d_type = "bfloat16"
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bfloat16,
        "float64": np.float64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool": np.bool_,
        "fp8_e4m3fn": np.uint8,
        "fp8_e5m2": np.uint8,
    }
    np_type = d_type_dict[d_type]
    
    # 生成输入数据
    input_k = np.ones((1, 10016, 2, 128)).astype(d_type_dict["bfloat16"])
    input_beta = np.ones((1, 10016, 8)).astype(d_type_dict["bfloat16"])
    input_g_cumsum = np.ones((1, 10016, 8)).astype(d_type_dict["float32"])
    input_chunk_offsets = np.ones((160)).astype(d_type_dict["int32"])
    attr_chunk_size = 64
    
    # 计算 golden 数据
    golden = impl(input_k, input_beta, input_g_cumsum, input_chunk_offsets, attr_chunk_size)
    
    # 保存数据到文件
    input_k.astype(d_type_dict["bfloat16"]).tofile(f"{d_type}_input_chunk_scaled_dot_kkt_k.bin")
    input_beta.astype(d_type_dict["bfloat16"]).tofile(f"{d_type}_input_chunk_scaled_dot_kkt_beta.bin")
    input_g_cumsum.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_chunk_scaled_dot_kkt_g_cumsum.bin")
    input_chunk_offsets.astype(d_type_dict["int32"]).tofile(f"{d_type}_input_chunk_scaled_dot_kkt_chunk_offsets.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float32_golden_chunk_scaled_dot_kkt.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float32"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float32"]).tofile("float32_golden_chunk_scaled_dot_kkt.bin")
    
    print(f"生成完成: dtype={d_type}")
