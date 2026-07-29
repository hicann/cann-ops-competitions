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
  - aclshmemx_malloc / aclshmemx_calloc / aclshmemx_align / aclshmemx_free
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.env import bootstrap_torch_dist, require_api
from common.assert_utils import check_eq, check_true, check_ne, skip_if
from common.shmem_fixture import ShmemSession
import shmem as ash

ALLOC_SIZE = 4 * 1024 * 1024


def _mem_type_device():
    if hasattr(ash, "MemType"):
        return ash.MemType.DEVICE_SIDE
    return None


def test_x_malloc_calloc_align_free(pe, world_size):
    x_malloc = require_api(ash, "aclshmemx_malloc")
    x_calloc = require_api(ash, "aclshmemx_calloc")
    x_align = require_api(ash, "aclshmemx_align")
    x_free = require_api(ash, "aclshmemx_free")
    mem_type = _mem_type_device()

    with ShmemSession(pe, world_size):
        # normal path
        ptr = x_malloc(ALLOC_SIZE, mem_type) if mem_type is not None else x_malloc(ALLOC_SIZE)
        check_ne(ptr, 0, "aclshmemx_malloc")
        check_ne(ptr, None, "aclshmemx_malloc")

        # size=0 boundary
        z = x_malloc(0, mem_type) if mem_type is not None else x_malloc(0)
        check_true(z in (0, None), "aclshmemx_malloc(0) should return null/0")

        cptr = x_calloc(1024, 4, mem_type) if mem_type is not None else x_calloc(1024, 4)
        check_ne(cptr, 0, "aclshmemx_calloc")

        aptr = x_align(256, ALLOC_SIZE, mem_type) if mem_type is not None else x_align(256, ALLOC_SIZE)
        check_ne(aptr, 0, "aclshmemx_align")
        check_eq(aptr % 256, 0, "align 256")

        if mem_type is not None:
            x_free(ptr, mem_type)
            x_free(cptr, mem_type)
            x_free(aptr, mem_type)
        else:
            x_free(ptr)
            x_free(cptr)
            x_free(aptr)

        print(f"pe[{pe}] heap mem_type malloc/calloc/align/free ok")


def test_host_side_if_supported(pe, world_size):
    """HOST_SIDE mem_type is optional; skip when MemType or HOST_SIDE unavailable."""
    skip_if(not hasattr(ash, "MemType"), "MemType enum not exported")
    skip_if(not hasattr(ash.MemType, "HOST_SIDE"), "HOST_SIDE not available")
    x_malloc = require_api(ash, "aclshmemx_malloc")
    x_free = require_api(ash, "aclshmemx_free")

    with ShmemSession(pe, world_size):
        ptr = x_malloc(4096, ash.MemType.HOST_SIDE)
        check_ne(ptr, 0, "HOST_SIDE malloc")
        x_free(ptr, ash.MemType.HOST_SIDE)
        print(f"pe[{pe}] HOST_SIDE malloc/free ok")


def run_tests(pe, world_size):
    test_x_malloc_calloc_align_free(pe, world_size)
    test_host_side_if_supported(pe, world_size)


if __name__ == "__main__":
    pe, world_size = bootstrap_torch_dist()
    run_tests(pe, world_size)
    print("test_heap_mem_type.py running success!")
