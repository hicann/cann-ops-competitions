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
Functional tests for shmem.core high-level wrappers (task pending):
  - barrier / barrier_all / sync / sync_all (+ on_stream)
  - multi_instance (ctx_get/set + mem_type malloc)
  - handle_wait
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.env import bootstrap_torch_dist, require_api
from common.assert_utils import check_true, skip_if
from common.shmem_fixture import DEFAULT_HEAP_SIZE
import shmem as ash
import shmem.core as core


def test_core_barrier_sync(pe, world_size):
    barrier = require_api(core, "barrier")
    barrier_all = require_api(core, "barrier_all") if hasattr(core, "barrier_all") else None
    sync = require_api(core, "sync") if hasattr(core, "sync") else None

    # core.init via unique_id
    import torch
    import torch.distributed as dist

    ash.set_conf_store_tls(False, "")
    uid_size = 512
    tensor = torch.zeros(uid_size, dtype=torch.uint8)
    if pe == 0:
        unique_id = core.get_unique_id()
        tensor = torch.tensor(list(unique_id), dtype=torch.uint8)
    dist.broadcast(tensor, src=0)
    unique_id = bytes(tensor.tolist())
    core.init(rank=pe, nranks=world_size, mem_size=DEFAULT_HEAP_SIZE,
              uid=unique_id, initializer_method="uid")

    barrier(0)
    if barrier_all is not None:
        barrier_all()
    if sync is not None:
        sync(0)

    if hasattr(core, "barrier_on_stream"):
        import torch_npu  # noqa: F401
        stream = torch.npu.Stream()
        stream_ptr = stream.npu_stream if hasattr(stream, "npu_stream") else 0
        core.barrier_on_stream(0, stream_ptr)
        stream.synchronize()

    core.finalize()
    print(f"pe[{pe}] core barrier/sync wrappers ok")


def test_core_multi_instance(pe, world_size):
    skip_if(not hasattr(core, "instance_ctx_set") and not hasattr(ash, "aclshmemx_instance_ctx_set"),
            "core multi_instance wrappers not implemented")
    # Prefer core namespace if present
    mod = core if hasattr(core, "instance_ctx_set") else ash
    set_fn = require_api(mod, "instance_ctx_set" if mod is core else "aclshmemx_instance_ctx_set")
    get_fn = require_api(mod, "instance_ctx_get" if mod is core else "aclshmemx_instance_ctx_get")

    ash.set_conf_store_tls(False, "")
    attributes = ash.InitAttr()
    attributes.my_rank = pe
    attributes.n_ranks = world_size
    attributes.local_mem_size = DEFAULT_HEAP_SIZE
    attributes.ip_port = "tcp://127.0.0.1:8767"
    attributes.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    ret = ash.aclshmem_init(attributes)
    check_true(ret == 0, "init for multi_instance")

    ctx = get_fn()
    check_true(ctx not in (0, None), "ctx_get")
    set_fn(0)

    if hasattr(ash, "aclshmemx_malloc"):
        ptr = ash.aclshmemx_malloc(4096, ash.MemType.DEVICE_SIDE if hasattr(ash, "MemType") else None)
        if ptr not in (0, None):
            ash.aclshmemx_free(ptr, ash.MemType.DEVICE_SIDE if hasattr(ash, "MemType") else None)

    ash.aclshmem_finalize()
    print(f"pe[{pe}] core multi_instance ok")


def test_core_handle_wait(pe, world_size):
    skip_if(world_size < 2, "need >= 2 PEs")
    skip_if(not hasattr(core, "handle_wait"), "core.handle_wait not implemented")
    hw = require_api(core, "handle_wait")
    # smoke: binding exists and is callable
    check_true(callable(hw), "core.handle_wait callable")
    print(f"pe[{pe}] core.handle_wait symbol ok (full RMA path covered in test_handle_wait.py)")


def run_tests(pe, world_size):
    test_core_barrier_sync(pe, world_size)
    test_core_multi_instance(pe, world_size)
    test_core_handle_wait(pe, world_size)


if __name__ == "__main__":
    pe, world_size = bootstrap_torch_dist()
    run_tests(pe, world_size)
    print("test_core_wrappers.py running success!")
