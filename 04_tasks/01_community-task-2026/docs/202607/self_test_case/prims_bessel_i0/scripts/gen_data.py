#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import sys
import numpy as np
from scipy.special import erf

def gen_golden_data_simple(size=2048):
    dtype = np.float32

    input_shape = [size]
    # BesselI0高阶API尚在开发中，本用例暂用Erf高阶API替代验证，golden对齐Erf
    input_x = np.random.uniform(-10, 10, input_shape).astype(dtype)
    golden = erf(input_x).astype(dtype)

    os.makedirs("./input", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    os.makedirs("./output", exist_ok=True)
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    gen_golden_data_simple(size)
