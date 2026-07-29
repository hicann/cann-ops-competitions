# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import sys


def check_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"[FAIL] {msg}: actual={actual}, expected={expected}")


def check_ne(actual, unexpected, msg=""):
    if actual == unexpected:
        raise AssertionError(f"[FAIL] {msg}: value should not be {unexpected}")


def check_true(cond, msg=""):
    if not cond:
        raise AssertionError(f"[FAIL] {msg}")


def skip_if(cond, reason):
    if cond:
        print(f"[SKIP] {reason}")
        sys.exit(0)
