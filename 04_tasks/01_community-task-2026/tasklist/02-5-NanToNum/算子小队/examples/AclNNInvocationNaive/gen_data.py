#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import sys
import torch
import numpy as np
import tensorflow as tf
bfp16 = tf.bfloat16.as_numpy_dtype

testcase = {
    "test_1" : {"shape":[1], "dtype":np.float16},
    "test_2" : {"shape":[1], "dtype":np.float32},
    "test_3" : {"shape":[1], "dtype":np.float32},
    "test_4" : {"shape":[7], "dtype":np.float32},
    "test_5" : {"shape":[8, 1], "dtype":np.float32},
    "test_6" : {"shape":[8], "dtype":np.float32},
    "test_7" : {"shape":[16, 1], "dtype":np.float32},
    "test_8" : {"shape":[19], "dtype":bfp16},
    "test_9" : {"shape":[20], "dtype":bfp16},
    "test_10" : {"shape":[20], "dtype":np.float32},
    "test_11" : {"shape":[20], "dtype":np.float32},
    "test_12" : {"shape":[9,15], "dtype":np.float16},
    "test_13" : {"shape":[8,17], "dtype":np.float16},
    "test_14" : {"shape":[7,21], "dtype":np.float16},
    "test_15" : {"shape":[255], "dtype":np.float16},
    "test_16" : {"shape":[20,20], "dtype":np.float16},
    "test_17" : {"shape":[256,8], "dtype":np.float16},
    "test_18" : {"shape":[19,9,16], "dtype":np.float16},
    "test_19" : {"shape":[20,20,7], "dtype":np.float16},
    "test_20" : {"shape":[257, 15], "dtype":np.float16},
    "test_21" : {"shape":[9,1,9,8,7], "dtype":np.float16},
    "test_22" : {"shape":[1, 20, 16, 17], "dtype":np.float16},
    "test_23" : {"shape":[1, 21, 1, 20, 19], "dtype":np.float16},
    "test_24" : {"shape":[16, 1, 7, 9, 21], "dtype":np.float16},
    "test_25" : {"shape":[17, 7, 15, 1, 161], "dtype":np.float16},
    "test_26" : {"shape":[7, 257, 19], "dtype":np.float16},
    "test_27" : {"shape":[9, 15, 255], "dtype":np.float16},
    "test_28" : {"shape":[9, 1, 255, 17, 1], "dtype":np.float16},
    "test_29" : {"shape":[17, 255, 15, 1], "dtype":np.float16},
    "test_30" : {"shape":[1, 16, 257, 21], "dtype":np.float16},
    "test_31" : {"shape":[131073], "dtype":np.float16},
    "test_32" : {"shape":[9, 9, 15, 19, 7], "dtype":np.float16},
    "test_33" : {"shape":[9, 7, 20, 21, 20], "dtype":np.float16},
    "test_34" : {"shape":[19, 17, 9, 255], "dtype":np.float16},
    "test_35" : {"shape":[15, 1, 8, 1, 7, 7, 8, 21], "dtype":np.float16},
    "test_36" : {"shape":[16, 8, 1, 8, 8, 8, 1, 21], "dtype":np.float16},
    "test_37" : {"shape":[19, 16, 16, 20, 19], "dtype":np.float16},
    "test_38" : {"shape":[16, 131073], "dtype":np.float16},
    "test_39" : {"shape":[15, 255, 19, 7, 8], "dtype":np.float16},
    "test_40" : {"shape":[20, 19, 15, 1, 7, 1, 9, 16], "dtype":np.float16},
    "test_41" : {"shape":[20, 256, 9, 8, 16], "dtype":np.float16},
    "test_42" : {"shape":[1, 8, 21, 20, 8, 16, 16], "dtype":np.float16},
    "test_43" : {"shape":[20, 7, 17, 15, 255], "dtype":np.float16},
    "test_44" : {"shape":[21, 256, 255, 7], "dtype":np.float16},
    "test_45" : {"shape":[21, 19, 21, 7, 9, 21], "dtype":np.float16},
    "test_46" : {"shape":[21, 7, 19, 257, 1, 17], "dtype":np.float16},
    "test_47" : {"shape":[21, 8, 131073, 1], "dtype":np.float16},
    "test_48" : {"shape":[20, 9, 131073, 1], "dtype":np.float16},
    "test_49" : {"shape":[17, 16, 20, 20, 20, 16], "dtype":np.float16},
    "test_50" : {"shape":[1, 17, 1, 256, 15, 9, 9, 7], "dtype":np.float16},
}

def gen_golden_data_simple(shape, dtype):
    dtype = dtype
    input_shape = shape
    x = np.random.uniform(-1, 1, input_shape).astype(dtype)
    if input_shape[0] > 3:
        x.flat[0] = np.nan
        x.flat[1] = np.inf
        x.flat[2] = -np.inf
    nanIn = np.float32(0.0)
    posIn = np.float32(1)
    negIn = np.float32(-1)
    if dtype == bfp16:
        x = x.astype(np.float32)
    golden = np.nan_to_num(x, nan=nanIn, posinf = posIn, neginf = negIn).astype(dtype)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.astype(dtype).tofile("./input/input_x.bin")
    golden.astype(dtype).tofile("./output/golden.bin")

if __name__ == "__main__":
    case_name = sys.argv[1]
    case_list = testcase.get(case_name)
    gen_golden_data_simple(case_list.get("shape"), case_list.get("dtype"))
