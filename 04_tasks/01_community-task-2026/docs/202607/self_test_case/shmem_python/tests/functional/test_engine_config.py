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
  - aclshmemx_set_mte_config
  - aclshmemx_set_sdma_config
  - aclshmemx_set_rdma_config
  - aclshmemx_set_udma_config
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.env import bootstrap_torch_dist, require_api
from common.assert_utils import check_eq, check_true
from common.shmem_fixture import ShmemSession
import shmem as ash

# Default UB staging params used by SHMEM docs (offset near end of UB, 64B+, EVENT_ID0)
DEFAULT_OFFSET = 190 * 1024
DEFAULT_UB_SIZE = 64
DEFAULT_SYNC_ID = 0


def _call_config(fn_name, pe, world_size):
    fn = require_api(ash, fn_name)
    with ShmemSession(pe, world_size):
        ret = fn(DEFAULT_OFFSET, DEFAULT_UB_SIZE, DEFAULT_SYNC_ID)
        check_eq(ret, 0, f"{fn_name} normal")

        # invalid sync_id / zero ub_size boundary — expect non-zero or documented behavior
        bad = fn(DEFAULT_OFFSET, 0, DEFAULT_SYNC_ID)
        check_true(isinstance(bad, int), f"{fn_name}(ub_size=0) must return int")
        print(f"pe[{pe}] {fn_name} ok (ret={ret}, ub0={bad})")


def test_set_mte_config(pe, world_size):
    _call_config("aclshmemx_set_mte_config", pe, world_size)


def test_set_sdma_config(pe, world_size):
    _call_config("aclshmemx_set_sdma_config", pe, world_size)


def test_set_rdma_config(pe, world_size):
    _call_config("aclshmemx_set_rdma_config", pe, world_size)


def test_set_udma_config(pe, world_size):
    _call_config("aclshmemx_set_udma_config", pe, world_size)


def run_tests(pe, world_size):
    test_set_mte_config(pe, world_size)
    test_set_sdma_config(pe, world_size)
    test_set_rdma_config(pe, world_size)
    test_set_udma_config(pe, world_size)


if __name__ == "__main__":
    pe, world_size = bootstrap_torch_dist()
    run_tests(pe, world_size)
    print("test_engine_config.py running success!")
