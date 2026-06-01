#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import array
import math
import sys


def read_float32(path):
    data = array.array("f")
    with open(path, "rb") as input_file:
        data.fromfile(input_file, 12)
    return list(data)


def main():
    output = read_float32("float32_output_t_roll.bin")
    golden = read_float32("float32_golden_t_roll.bin")
    for index, (actual, expected) in enumerate(zip(output, golden)):
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
            print(f"FAILED: index={index}, output={actual}, golden={expected}")
            return 1
    print("PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
