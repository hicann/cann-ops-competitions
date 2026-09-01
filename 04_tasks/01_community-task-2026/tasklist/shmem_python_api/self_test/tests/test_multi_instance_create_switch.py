import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

import shmem as ash


HEAP_SIZE = 64 * 1024 * 1024
INSTANCE_A = 1
INSTANCE_B = 2


def make_attr(rank, world_size, instance_id):
    attr = ash.InitAttr()

    attr.my_rank = rank
    attr.n_ranks = world_size
    attr.local_mem_size = HEAP_SIZE

    # Multi-instance default mode:
    # input port = 0; runtime uses
    # SHMEM_INSTANCE_PORT_RANGE + instance_id.
    attr.ip_port = "tcp://127.0.0.1:0"

    attr.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    attr.instance_id = instance_id

    return attr


def finalize_instance(instance_id):
    # Select the target instance, then finalize the current instance.
    ret = ash.aclshmemx_instance_ctx_set(instance_id)
    if ret != 0:
        return ret
    ash.aclshmem_finalize()
    return 0


def create_instance(rank, world_size, instance_id):
    attr = make_attr(
        rank,
        world_size,
        instance_id,
    )

    ret = ash.aclshmem_init(attr)

    print(
        f"PE{rank}: INIT id={instance_id} ret={ret}",
        flush=True,
    )

    if ret != 0:
        raise RuntimeError(
            f"PE{rank}: create instance "
            f"{instance_id} failed ret={ret}"
        )

    ctx = ash.aclshmemx_instance_ctx_get()

    if ctx in (0, None):
        raise AssertionError(
            f"PE{rank}: instance {instance_id} "
            "returned invalid context"
        )

    if ash.my_pe() != rank:
        raise AssertionError(
            f"PE{rank}: my_pe mismatch "
            f"after creating instance {instance_id}"
        )

    if ash.pe_count() != world_size:
        raise AssertionError(
            f"PE{rank}: pe_count mismatch "
            f"after creating instance {instance_id}"
        )

    print(
        f"PE{rank}: INSTANCE_CREATE_PASS "
        f"id={instance_id} ctx={ctx}",
        flush=True,
    )

    return ctx


def switch_and_check(rank, world_size, instance_id, expected_ctx):
    ret = ash.aclshmemx_instance_ctx_set(instance_id)

    if ret != 0:
        raise RuntimeError(
            f"PE{rank}: ctx_set({instance_id}) "
            f"failed ret={ret}"
        )

    ctx = ash.aclshmemx_instance_ctx_get()

    if ctx != expected_ctx:
        raise AssertionError(
            f"PE{rank}: ctx mismatch for "
            f"instance {instance_id}: "
            f"actual={ctx}, expected={expected_ctx}"
        )

    if ash.my_pe() != rank:
        raise AssertionError(
            f"PE{rank}: my_pe polluted "
            f"after switch to {instance_id}"
        )

    if ash.pe_count() != world_size:
        raise AssertionError(
            f"PE{rank}: pe_count polluted "
            f"after switch to {instance_id}"
        )

    print(
        f"PE{rank}: CONTEXT_SWITCH_PASS "
        f"id={instance_id} ctx={ctx}",
        flush=True,
    )


def verify_switch_sequence(rank, world_size, ctx_a, ctx_b):
    sequence = (
        (INSTANCE_A, ctx_a),
        (INSTANCE_B, ctx_b),
        (INSTANCE_A, ctx_a),
    )
    for instance_id, expected_ctx in sequence:
        switch_and_check(
            rank,
            world_size,
            instance_id,
            expected_ctx,
        )
        dist.barrier()


def cleanup_instances(rank):
    for name, instance_id in (
        ("B", INSTANCE_B),
        ("A", INSTANCE_A),
    ):
        ret = finalize_instance(instance_id)
        print(
            f"PE{rank}: FINALIZE_{name} ret={ret}",
            flush=True,
        )
        if ret != 0:
            raise RuntimeError(
                f"PE{rank}: finalize {name} failed ret={ret}"
            )


def prepare_runtime():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(
            f"Expected 2 PEs, got {world_size}"
        )

    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise RuntimeError(
            f"set_conf_store_tls failed ret={ret}"
        )
    return rank, world_size


def main():
    rank, world_size = prepare_runtime()

    ctx_a = create_instance(
        rank,
        world_size,
        INSTANCE_A,
    )

    dist.barrier()

    ctx_b = create_instance(
        rank,
        world_size,
        INSTANCE_B,
    )

    dist.barrier()

    if ctx_a == ctx_b:
        raise AssertionError(
            f"PE{rank}: instance A and B "
            "returned the same context"
        )

    verify_switch_sequence(
        rank,
        world_size,
        ctx_a,
        ctx_b,
    )

    print(
        f"PE{rank}: MULTI_INSTANCE_SWITCH_PASS "
        f"ctx_a={ctx_a} ctx_b={ctx_b}",
        flush=True,
    )

    cleanup_instances(rank)
    dist.barrier()

    print(
        f"PE{rank}: MULTI_INSTANCE_CREATE_SWITCH_PASS",
        flush=True,
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
                "MULTI_INSTANCE_CREATE_SWITCH_OVERALL_PASS",
                flush=True,
            )

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
