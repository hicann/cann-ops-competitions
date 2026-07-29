# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Functional tests for:
  - aclshmemx_get_prof
  - aclshmemx_show_prof
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.env import bootstrap_torch_dist, require_api
from common.assert_utils import check_true
from common.shmem_fixture import ShmemSession
import shmem as ash


def test_show_prof(pe, world_size):
    show_prof = require_api(ash, "aclshmemx_show_prof")
    with ShmemSession(pe, world_size):
        # should not crash; may print to stdout
        show_prof()
        print(f"pe[{pe}] show_prof ok")


def test_get_prof(pe, world_size):
    get_prof = require_api(ash, "aclshmemx_get_prof")
    with ShmemSession(pe, world_size):
        # verbose=False: retrieve without console dump when supported
        result = get_prof(None, False)
        # binding may return list/tuple/None depending on out-param design
        check_true(result is None or isinstance(result, (list, tuple)),
                   "get_prof return type")
        # verbose=True equivalent to show_prof
        _ = get_prof(None, True)
        print(f"pe[{pe}] get_prof ok")


def run_tests(pe, world_size):
    test_show_prof(pe, world_size)
    test_get_prof(pe, world_size)


if __name__ == "__main__":
    pe, world_size = bootstrap_torch_dist()
    run_tests(pe, world_size)
    print("test_prof.py running success!")
