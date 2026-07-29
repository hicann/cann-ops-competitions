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
  - aclshmemx_handle_wait

Semantics: after Host/Device RMA on a stream, wait until team handle completes.
Requires at least putmem_on_stream (already bound) + handle_wait (task pending).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.env import bootstrap_torch_dist, require_api
from common.assert_utils import check_true, skip_if
from common.shmem_fixture import ShmemSession
import shmem as ash

MSG_SIZE = 4096
WORLD_TEAM = 0


def _make_handle():
    """Build aclshmem_handle_t-like object if Python binding exposes a helper/class."""
    if hasattr(ash, "Handle"):
        h = ash.Handle()
        if hasattr(h, "team_id"):
            h.team_id = WORLD_TEAM
        return h
    # fallback: plain int team id if binding accepts it
    return WORLD_TEAM


def test_handle_wait_after_put(pe, world_size):
    handle_wait = require_api(ash, "aclshmemx_handle_wait")
    put_on_stream = require_api(ash, "aclshmemx_putmem_on_stream")

    import torch
    import torch_npu  # noqa: F401

    with ShmemSession(pe, world_size):
        ptr = ash.aclshmem_malloc(MSG_SIZE)
        check_true(ptr not in (0, None), "malloc for handle_wait test")
        peer = (pe + 1) % world_size
        stream = torch.npu.Stream()
        stream_ptr = stream.npu_stream if hasattr(stream, "npu_stream") else 0

        with torch.npu.stream(stream):
            put_on_stream(ptr, ptr, MSG_SIZE, peer, stream_ptr)
            handle = _make_handle()
            handle_wait(handle, stream_ptr)
        stream.synchronize()
        ash.aclshmem_free(ptr)
        print(f"pe[{pe}] handle_wait after putmem_on_stream ok")


def run_tests(pe, world_size):
    skip_if(world_size < 2, "handle_wait RMA path needs >= 2 PEs")
    test_handle_wait_after_put(pe, world_size)


if __name__ == "__main__":
    pe, world_size = bootstrap_torch_dist()
    run_tests(pe, world_size)
    print("test_handle_wait.py running success!")
