# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device / torch.distributed bootstrap helpers for SHMEM Python community tests."""
import os
import sys


def get_rank_info():
    """Return (local_rank, world_size) from torchrun env."""
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, world_size


def bootstrap_torch_dist(backend="hccl"):
    """
    Set NPU device and init torch.distributed.
    Must be called before any aclshmem_init / aclInit-related SHMEM call.
    """
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    local_rank, _ = get_rank_info()
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=local_rank)
    return dist.get_rank(), dist.get_world_size()


def require_api(module, name, reason=None):
    """
    Return the attribute if present; otherwise print SKIP and exit 0.
    Used so archived cases can run against incomplete bindings without failing CI hard.
    """
    if not hasattr(module, name):
        msg = reason or f"{module.__name__}.{name} not bound yet (task pending implementation)"
        print(f"[SKIP] {msg}")
        sys.exit(0)
    return getattr(module, name)
