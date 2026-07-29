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
  - aclshmem_barrier / aclshmem_barrier_all
  - aclshmem_sync / aclshmem_sync_all
  - aclshmemx_barrier_on_stream / aclshmemx_barrier_all_on_stream
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.env import bootstrap_torch_dist, require_api
from common.assert_utils import check_eq
from common.shmem_fixture import ShmemSession
import shmem as ash

# WORLD team id is conventionally 0
WORLD_TEAM = 0


def test_barrier_and_sync(pe, world_size):
    barrier = require_api(ash, "aclshmem_barrier")
    barrier_all = require_api(ash, "aclshmem_barrier_all")
    sync = require_api(ash, "aclshmem_sync")
    sync_all = require_api(ash, "aclshmem_sync_all")

    with ShmemSession(pe, world_size):
        # collective: all PEs must call
        barrier(WORLD_TEAM)
        barrier_all()
        sync(WORLD_TEAM)
        sync_all()
        # second round to catch sticky state bugs
        barrier_all()
        sync_all()
        print(f"pe[{pe}] barrier/sync ok")


def test_barrier_on_stream(pe, world_size):
    barrier_on_stream = require_api(ash, "aclshmemx_barrier_on_stream")
    barrier_all_on_stream = require_api(ash, "aclshmemx_barrier_all_on_stream")

    import torch
    import torch_npu  # noqa: F401

    with ShmemSession(pe, world_size):
        stream = torch.npu.Stream()
        # pass stream as raw ptr / handle depending on binding convention
        stream_ptr = stream.npu_stream if hasattr(stream, "npu_stream") else int(stream.cuda_stream) if hasattr(stream, "cuda_stream") else 0
        with torch.npu.stream(stream):
            barrier_on_stream(WORLD_TEAM, stream_ptr)
            barrier_all_on_stream(stream_ptr)
        stream.synchronize()
        print(f"pe[{pe}] barrier_on_stream ok")


def run_tests(pe, world_size):
    test_barrier_and_sync(pe, world_size)
    test_barrier_on_stream(pe, world_size)


if __name__ == "__main__":
    pe, world_size = bootstrap_torch_dist()
    run_tests(pe, world_size)
    print("test_barrier_sync.py running success!")
