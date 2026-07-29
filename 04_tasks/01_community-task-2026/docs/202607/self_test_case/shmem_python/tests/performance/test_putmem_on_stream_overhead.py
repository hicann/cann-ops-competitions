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
Performance smoke: measure Python aclshmemx_putmem_on_stream latency vs a simple baseline.

Acceptance target (task book): Python Host RMA overhead vs same-path C++ <= 5%.
This script documents the measurement method. If C++ reference binary is unavailable,
it still reports Python-side avg latency and exits 0 with a note.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.env import bootstrap_torch_dist, require_api
from common.assert_utils import check_true, skip_if
from common.shmem_fixture import ShmemSession
import shmem as ash

MSG_SIZE = 64 * 1024
ITERS = 100
WARMUP = 10
OVERHEAD_LIMIT = 0.05  # 5%


def _stream_ptr(stream):
    return stream.npu_stream if hasattr(stream, "npu_stream") else 0


def bench_python_put(pe, world_size):
    put = require_api(ash, "aclshmemx_putmem_on_stream")
    import torch
    import torch_npu  # noqa: F401

    with ShmemSession(pe, world_size):
        ptr = ash.aclshmem_malloc(MSG_SIZE)
        check_true(ptr not in (0, None), "malloc")
        peer = (pe + 1) % world_size
        stream = torch.npu.Stream()
        sp = _stream_ptr(stream)

        for _ in range(WARMUP):
            put(ptr, ptr, MSG_SIZE, peer, sp)
        stream.synchronize()

        t0 = time.perf_counter()
        for _ in range(ITERS):
            put(ptr, ptr, MSG_SIZE, peer, sp)
        stream.synchronize()
        t1 = time.perf_counter()

        ash.aclshmem_free(ptr)
        avg_us = (t1 - t0) * 1e6 / ITERS
        print(f"pe[{pe}] python putmem_on_stream avg={avg_us:.3f} us "
              f"(size={MSG_SIZE}, iters={ITERS})")
        return avg_us


def compare_with_cpp_baseline(py_avg_us):
    """
    Optional: set SHMEM_CPP_PUT_BASELINE_US to the C++ reference avg latency (us).
    If unset, skip overhead check and only print guidance.
    """
    baseline = os.environ.get("SHMEM_CPP_PUT_BASELINE_US")
    if not baseline:
        print("[NOTE] Set SHMEM_CPP_PUT_BASELINE_US=<cpp_avg_us> to enforce <=5% overhead check.")
        print("[NOTE] C++ reference: examples/rdma_perftest or host put-on-stream microbench.")
        return
    cpp_us = float(baseline)
    overhead = (py_avg_us - cpp_us) / cpp_us
    print(f"overhead={(overhead * 100):.2f}% (limit {OVERHEAD_LIMIT * 100}%)")
    check_true(overhead <= OVERHEAD_LIMIT,
               f"Python overhead {overhead:.2%} exceeds {OVERHEAD_LIMIT:.0%}")


def run_tests(pe, world_size):
    skip_if(world_size < 2, "need >= 2 PEs for put bandwidth/latency smoke")
    avg = bench_python_put(pe, world_size)
    if pe == 0:
        compare_with_cpp_baseline(avg)


if __name__ == "__main__":
    pe, world_size = bootstrap_torch_dist()
    run_tests(pe, world_size)
    print("test_putmem_on_stream_overhead.py running success!")
