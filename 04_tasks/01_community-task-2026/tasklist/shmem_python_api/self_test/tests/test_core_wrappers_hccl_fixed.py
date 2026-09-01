import sys

TEST_ROOT = (
    "/workspace/cann-ops-competitions/04_tasks/01_community-task-2026/"
    "docs/202607/self_test_case/shmem_python/tests"
)
sys.path.insert(0, TEST_ROOT)

from common.env import bootstrap_torch_dist, require_api
from common.assert_utils import check_true, skip_if
from common.shmem_fixture import DEFAULT_HEAP_SIZE

import shmem as ash
import shmem.core as core


def _broadcast_unique_id_hccl(pe):
    import torch
    import torch.distributed as dist

    # HCCL requires NPU tensors. First broadcast the real UID length.
    if pe == 0:
        unique_id = bytes(core.get_unique_id())
        uid_len = len(unique_id)
        print(f"pe[{pe}] unique_id length={uid_len}")

        len_tensor = torch.tensor(
            [uid_len],
            dtype=torch.int32,
            device="npu",
        )
    else:
        unique_id = None
        len_tensor = torch.zeros(
            1,
            dtype=torch.int32,
            device="npu",
        )

    dist.broadcast(len_tensor, src=0)

    uid_len = int(len_tensor.cpu().item())
    check_true(uid_len > 0, "unique_id length > 0")

    # Then broadcast the UID bytes themselves on NPU.
    if pe == 0:
        uid_tensor = torch.tensor(
            list(unique_id),
            dtype=torch.uint8,
            device="npu",
        )
    else:
        uid_tensor = torch.zeros(
            uid_len,
            dtype=torch.uint8,
            device="npu",
        )

    dist.broadcast(uid_tensor, src=0)

    result = bytes(uid_tensor.cpu().tolist())
    check_true(len(result) == uid_len, "broadcast unique_id length")

    print(f"pe[{pe}] unique_id broadcast ok, length={uid_len}")

    return result


def _check_barrier_on_stream(pe):
    if not hasattr(core, "barrier_on_stream"):
        return

    import torch
    import torch_npu  # noqa: F401

    stream = torch.npu.Stream()
    stream_ptr = (
        stream.npu_stream
        if hasattr(stream, "npu_stream")
        else 0
    )
    core.barrier_on_stream(0, stream_ptr)
    stream.synchronize()
    print(f"pe[{pe}] core.barrier_on_stream ok")


def test_core_barrier_sync(pe, world_size):
    barrier = require_api(core, "barrier")

    barrier_all = (
        require_api(core, "barrier_all")
        if hasattr(core, "barrier_all")
        else None
    )

    sync = (
        require_api(core, "sync")
        if hasattr(core, "sync")
        else None
    )

    ash.set_conf_store_tls(False, "")

    unique_id = _broadcast_unique_id_hccl(pe)

    core.init(
        rank=pe,
        nranks=world_size,
        mem_size=DEFAULT_HEAP_SIZE,
        uid=unique_id,
        initializer_method="uid",
    )

    print(f"pe[{pe}] core.init ok")

    barrier(0)

    if barrier_all is not None:
        barrier_all()

    if sync is not None:
        sync(0)

    _check_barrier_on_stream(pe)

    core.finalize()

    print(f"pe[{pe}] core barrier/sync wrappers ok")


def _check_multi_instance_allocation():
    if not hasattr(ash, "aclshmemx_malloc"):
        return

    mem_type = (
        ash.MemType.DEVICE_SIDE
        if hasattr(ash, "MemType")
        else None
    )
    ptr = ash.aclshmemx_malloc(4096, mem_type)
    if ptr not in (0, None):
        ash.aclshmemx_free(ptr, mem_type)


def test_core_multi_instance(pe, world_size):
    skip_if(
        not hasattr(core, "instance_ctx_set")
        and not hasattr(ash, "aclshmemx_instance_ctx_set"),
        "core multi_instance wrappers not implemented",
    )

    mod = (
        core
        if hasattr(core, "instance_ctx_set")
        else ash
    )

    set_fn = require_api(
        mod,
        "instance_ctx_set"
        if mod is core
        else "aclshmemx_instance_ctx_set",
    )

    get_fn = require_api(
        mod,
        "instance_ctx_get"
        if mod is core
        else "aclshmemx_instance_ctx_get",
    )

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

    _check_multi_instance_allocation()

    ash.aclshmem_finalize()

    print(f"pe[{pe}] core multi_instance ok")


def test_core_handle_wait(pe, world_size):
    skip_if(
        world_size < 2,
        "need >= 2 PEs",
    )

    skip_if(
        not hasattr(core, "handle_wait"),
        "core.handle_wait not implemented",
    )

    hw = require_api(
        core,
        "handle_wait",
    )

    check_true(
        callable(hw),
        "core.handle_wait callable",
    )

    print(
        f"pe[{pe}] core.handle_wait symbol ok "
        "(full RMA path requires ROCE environment)"
    )


def run_tests(pe, world_size):
    test_core_barrier_sync(
        pe,
        world_size,
    )

    test_core_multi_instance(
        pe,
        world_size,
    )

    test_core_handle_wait(
        pe,
        world_size,
    )


if __name__ == "__main__":
    pe, world_size = bootstrap_torch_dist()

    run_tests(
        pe,
        world_size,
    )

    print(
        "test_core_wrappers_hccl_fixed.py "
        "running success!"
    )
