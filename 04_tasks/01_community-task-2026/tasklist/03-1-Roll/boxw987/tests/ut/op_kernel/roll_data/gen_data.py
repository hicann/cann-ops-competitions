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
import os


def write_float32(path, values):
    data = array.array("f", values)
    with open(path, "wb") as output:
        data.tofile(output)


def main():
    for file_name in os.listdir("."):
        if file_name.endswith(".bin"):
            os.remove(file_name)

    values = [float(i) for i in range(12)]
    golden = values[9:12] + values[0:9]
    write_float32("float32_input_t_roll.bin", values)
    write_float32("float32_golden_t_roll.bin", golden)


if __name__ == "__main__":
    main()
