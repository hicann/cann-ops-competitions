import os
import ctypes

import acl
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

import shmem as ash


HEAP_SIZE = 64 * 1024 * 1024
ALLOC_SIZE = 4096
DATA_SIZE = 256

INSTANCE_A = 1
INSTANCE_B = 2

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def h2d(addr, data):
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)

    ret = acl.rt.memcpy(
        addr,
        len(data),
        ctypes.addressof(buf),
        len(data),
        ACL_MEMCPY_HOST_TO_DEVICE,
    )

    if ret != 0:
        raise RuntimeError(
            f"H2D memcpy failed ret={ret}"
        )


def d2h(addr, size):
    buf = (ctypes.c_ubyte * size)()

    ret = acl.rt.memcpy(
        ctypes.addressof(buf),
        size,
        addr,
        size,
        ACL_MEMCPY_DEVICE_TO_HOST,
    )

    if ret != 0:
        raise RuntimeError(
            f"D2H memcpy failed ret={ret}"
        )

    return bytes(buf)


def make_attr(rank, world_size, instance_id):
    attr = ash.InitAttr()

    attr.my_rank = rank
    attr.n_ranks = world_size
    attr.local_mem_size = HEAP_SIZE
    attr.ip_port = "tcp://127.0.0.1:0"
    attr.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    attr.instance_id = instance_id

    return attr


def create_instance(rank, world_size, instance_id):
    attr = make_attr(
        rank,
        world_size,
        instance_id,
    )

    ret = ash.aclshmem_init(attr)

    if ret != 0:
        raise RuntimeError(
            f"PE{rank}: init instance "
            f"{instance_id} failed ret={ret}"
        )

    ctx = ash.aclshmemx_instance_ctx_get()

    if ctx in (0, None):
        raise AssertionError(
            f"PE{rank}: bad ctx for "
            f"instance {instance_id}"
        )

    print(
        f"PE{rank}: CREATE_PASS "
        f"id={instance_id} ctx={ctx}",
        flush=True,
    )

    return ctx


def select_instance(rank, instance_id, expected_ctx):
    ret = ash.aclshmemx_instance_ctx_set(
        instance_id
    )

    if ret != 0:
        raise RuntimeError(
            f"PE{rank}: ctx_set({instance_id}) "
            f"failed ret={ret}"
        )

    ctx = ash.aclshmemx_instance_ctx_get()

    if ctx != expected_ctx:
        raise AssertionError(
            f"PE{rank}: ctx mismatch "
            f"id={instance_id}"
        )


def finalize_instance(instance_id):
    ret = ash.aclshmemx_instance_ctx_set(instance_id)
    if ret != 0:
        return ret
    ash.aclshmem_finalize()
    return 0


def _prepare_runtime():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != 2:
        raise RuntimeError(
            f"need 2 PEs, got {world_size}"
        )

    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise RuntimeError(
            f"disable TLS failed ret={ret}"
        )

    return rank, world_size


def _create_heap_state(rank, world_size, instance_id):
    ctx = create_instance(
        rank,
        world_size,
        instance_id,
    )

    ptr = ash.aclshmemx_malloc(
        ALLOC_SIZE,
        ash.MemType.DEVICE_SIDE,
    )
    if ptr in (0, None):
        raise RuntimeError(
            f"PE{rank}: malloc failed "
            f"for instance {instance_id}"
        )

    if instance_id == INSTANCE_A:
        base = 17
        step = 29
        label = "A"
    else:
        base = 193
        step = 37
        label = "B"

    pattern = bytes(
        ((base + rank * step + i) % 256)
        for i in range(DATA_SIZE)
    )
    h2d(ptr, pattern)

    if d2h(ptr, DATA_SIZE) != pattern:
        raise AssertionError(
            f"PE{rank}: initial {label} write mismatch"
        )

    print(
        f"PE{rank}: HEAP_{label}_WRITE_PASS "
        f"ptr={ptr}",
        flush=True,
    )

    return {
        "instance_id": instance_id,
        "ctx": ctx,
        "ptr": ptr,
        "pattern": pattern,
    }


def _check_heap(rank, state, marker, message):
    select_instance(
        rank,
        state["instance_id"],
        state["ctx"],
    )

    if d2h(state["ptr"], DATA_SIZE) != state["pattern"]:
        raise AssertionError(
            f"PE{rank}: {message}"
        )

    if marker:
        print(
            f"PE{rank}: {marker}",
            flush=True,
        )


def _allocate_extra(rank, label):
    ptr = ash.aclshmemx_malloc(
        ALLOC_SIZE,
        ash.MemType.DEVICE_SIDE,
    )
    if ptr in (0, None):
        raise RuntimeError(
            f"PE{rank}: {label} second malloc failed"
        )
    return ptr


def _exercise_switches(rank, state_a, state_b):
    _check_heap(
        rank,
        state_a,
        "HEAP_A_ISOLATION_PASS",
        "A heap corrupted after B creation/write",
    )
    ptr_a2 = _allocate_extra(rank, "A")
    dist.barrier()

    _check_heap(
        rank,
        state_b,
        "HEAP_B_ISOLATION_PASS",
        "B heap corrupted after switching to A",
    )
    ptr_b2 = _allocate_extra(rank, "B")
    dist.barrier()

    _check_heap(
        rank,
        state_a,
        None,
        "A data changed after repeated switch",
    )
    _check_heap(
        rank,
        state_b,
        None,
        "B data changed after repeated switch",
    )

    print(
        f"PE{rank}: REPEATED_SWITCH_DATA_PASS",
        flush=True,
    )

    return ptr_a2, ptr_b2


def _free_ptrs(*ptrs):
    for ptr in ptrs:
        ash.aclshmemx_free(
            ptr,
            ash.MemType.DEVICE_SIDE,
        )


def _finalize_checked(rank, label, instance_id):
    ret = finalize_instance(instance_id)
    if ret != 0:
        raise RuntimeError(
            f"PE{rank}: finalize {label} failed ret={ret}"
        )


def _cleanup_instances(rank, state_a, state_b, ptr_a2, ptr_b2):
    _free_ptrs(
        ptr_b2,
        state_b["ptr"],
    )
    _finalize_checked(
        rank,
        "B",
        INSTANCE_B,
    )

    _check_heap(
        rank,
        state_a,
        "DESTROY_B_KEEP_A_PASS",
        "A corrupted after B finalize",
    )

    _free_ptrs(
        ptr_a2,
        state_a["ptr"],
    )
    _finalize_checked(
        rank,
        "A",
        INSTANCE_A,
    )

    dist.barrier()

    print(
        f"PE{rank}: MULTI_INSTANCE_HEAP_ISOLATION_PASS",
        flush=True,
    )


def main():
    rank, world_size = _prepare_runtime()

    state_a = _create_heap_state(
        rank,
        world_size,
        INSTANCE_A,
    )
    dist.barrier()

    state_b = _create_heap_state(
        rank,
        world_size,
        INSTANCE_B,
    )

    if state_a["ctx"] == state_b["ctx"]:
        raise AssertionError(
            f"PE{rank}: A/B context collision"
        )

    dist.barrier()

    ptr_a2, ptr_b2 = _exercise_switches(
        rank,
        state_a,
        state_b,
    )

    _cleanup_instances(
        rank,
        state_a,
        state_b,
        ptr_a2,
        ptr_b2,
    )


if __name__ == "__main__":
    local_rank = int(
        os.environ.get("LOCAL_RANK", "0")
    )

    torch.npu.set_device(local_rank)

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
    )

    try:
        main()

        if dist.get_rank() == 0:
            print(
                "MULTI_INSTANCE_HEAP_ISOLATION_OVERALL_PASS",
                flush=True,
            )

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
