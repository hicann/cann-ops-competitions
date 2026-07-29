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
  - aclshmemx_set_attr_uniqueid_args
  - aclshmemx_instance_ctx_get
  - aclshmemx_instance_ctx_set
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.env import bootstrap_torch_dist, require_api
from common.assert_utils import check_eq, check_true, check_ne
from common.shmem_fixture import ShmemSession, DEFAULT_HEAP_SIZE
import shmem as ash


def test_set_attr_uniqueid_args(pe, world_size):
    """Export and call aclshmemx_set_attr_uniqueid_args (UID attr builder)."""
    fn = require_api(ash, "aclshmemx_set_attr_uniqueid_args")
    get_uid = require_api(ash, "aclshmem_get_unique_id")

    # rank0 creates uid bytes; all ranks need a valid uid for set_attr
    import torch
    import torch.distributed as dist

    uid_bytes = None
    if pe == 0:
        uid_bytes = get_uid()
        check_true(uid_bytes is not None and len(uid_bytes) > 0, "unique_id empty")
    # broadcast length then bytes via tensor
    if pe == 0:
        t = torch.tensor(list(uid_bytes), dtype=torch.uint8)
        n = torch.tensor([t.numel()], dtype=torch.int64)
    else:
        n = torch.zeros(1, dtype=torch.int64)
    dist.broadcast(n, src=0)
    if pe != 0:
        t = torch.zeros(int(n.item()), dtype=torch.uint8)
    dist.broadcast(t, src=0)
    uid_bytes = bytes(t.tolist())

    attr = ash.InitAttr()
    ret = fn(pe, world_size, DEFAULT_HEAP_SIZE, uid_bytes, attr)
    check_eq(ret, 0, "aclshmemx_set_attr_uniqueid_args")
    check_eq(attr.my_rank, pe, "attr.my_rank")
    check_eq(attr.n_ranks, world_size, "attr.n_ranks")
    print(f"pe[{pe}] set_attr_uniqueid_args ok")


def test_instance_ctx_get_set(pe, world_size):
    """Switch instance context after init; get must return non-null when initialized."""
    ctx_get = require_api(ash, "aclshmemx_instance_ctx_get")
    ctx_set = require_api(ash, "aclshmemx_instance_ctx_set")

    with ShmemSession(pe, world_size, instance_id=0) as _:
        ctx = ctx_get()
        check_ne(ctx, 0, "instance_ctx_get after init")
        check_ne(ctx, None, "instance_ctx_get after init")

        ret = ctx_set(0)
        check_eq(ret, 0, "instance_ctx_set(0)")

        # invalid / unused instance_id should fail or be rejected
        bad = ctx_set(999999)
        check_true(bad != 0, "instance_ctx_set(invalid) should fail")
        print(f"pe[{pe}] instance_ctx get/set ok")


def run_tests(pe, world_size):
    test_set_attr_uniqueid_args(pe, world_size)
    test_instance_ctx_get_set(pe, world_size)


if __name__ == "__main__":
    pe, world_size = bootstrap_torch_dist()
    run_tests(pe, world_size)
    print("test_init_multi_instance.py running success!")
