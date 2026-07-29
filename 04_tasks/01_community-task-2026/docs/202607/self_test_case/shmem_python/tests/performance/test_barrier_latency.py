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
Performance / robustness: barrier and barrier_on_stream must finish within timeout,
with no deadlock across PEs.
"""
import sys
import os
import time
import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.env import bootstrap_torch_dist, require_api
from common.assert_utils import check_true
from common.shmem_fixture import ShmemSession
import shmem as ash

WORLD_TEAM = 0
ITERS = 50
TIMEOUT_SEC = 60


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout("barrier test exceeded TIMEOUT_SEC")


def test_barrier_latency(pe, world_size):
    barrier_all = require_api(ash, "aclshmem_barrier_all")
    barrier_on_stream = require_api(ash, "aclshmemx_barrier_all_on_stream")

    import torch
    import torch_npu  # noqa: F401

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TIMEOUT_SEC)
    try:
        with ShmemSession(pe, world_size):
            t0 = time.perf_counter()
            for _ in range(ITERS):
                barrier_all()
            t1 = time.perf_counter()
            avg_host = (t1 - t0) * 1e6 / ITERS

            stream = torch.npu.Stream()
            sp = stream.npu_stream if hasattr(stream, "npu_stream") else 0
            t2 = time.perf_counter()
            for _ in range(ITERS):
                barrier_on_stream(sp)
            stream.synchronize()
            t3 = time.perf_counter()
            avg_stream = (t3 - t2) * 1e6 / ITERS

            check_true(avg_host > 0 and avg_stream > 0, "latency positive")
            print(f"pe[{pe}] barrier_all avg={avg_host:.3f} us; "
                  f"barrier_all_on_stream avg={avg_stream:.3f} us (iters={ITERS})")
    finally:
        signal.alarm(0)


def run_tests(pe, world_size):
    test_barrier_latency(pe, world_size)


if __name__ == "__main__":
    pe, world_size = bootstrap_torch_dist()
    run_tests(pe, world_size)
    print("test_barrier_latency.py running success!")
