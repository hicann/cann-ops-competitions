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
import numpy as np

LOSS = 1e-3 # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
MINIMUM = 10e-10
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

def verify_result(real_result, golden, dtype):
    real_result = np.fromfile(real_result, dtype=dtype) # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=dtype) # 从bin文件读取预期运算结果
    result = np.abs(real_result - golden) # 计算运算结果和预期结果偏差
    deno = np.maximum(np.abs(real_result), np.abs(golden))  # 获取最大值并组成新数组
    result_atol = np.less_equal(result, LOSS) # 计算绝对误差
    result_rtol = np.less_equal(result / np.add(deno, MINIMUM), LOSS) # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result.size * LOSS and \
           np.sum(result_atol == False) > real_result.size * LOSS: # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

if __name__ == '__main__':
    case_name = sys.argv[3]
    dtype = testcase.get(case_name).get("dtype", np.float16)
    verify_result(sys.argv[1],sys.argv[2], dtype)
