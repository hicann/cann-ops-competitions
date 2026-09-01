import os
import ctypes

import acl
import torch
import torch.distributed as dist

import shmem as ash
import shmem.core as core


MSG_SIZE = 256
HEAP_SIZE = 1024 * 1024 * 1024

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def h2d(addr, data: bytes):
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
            f"acl.rt.memcpy H2D failed, ret={ret}"
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
            f"acl.rt.memcpy D2H failed, ret={ret}"
        )

    return bytes(buf)


def broadcast_uid(rank):
    if rank == 0:
        uid = bytes(core.get_unique_id())
        size = torch.tensor(
            [len(uid)],
            dtype=torch.int64,
        )
    else:
        uid = b""
        size = torch.zeros(
            1,
            dtype=torch.int64,
        )

    dist.broadcast(size, src=0)
    uid_len = int(size.item())

    if rank == 0:
        tensor = torch.tensor(
            list(uid),
            dtype=torch.uint8,
        )
    else:
        tensor = torch.empty(
            uid_len,
            dtype=torch.uint8,
        )

    dist.broadcast(tensor, src=0)

    return bytes(tensor.tolist())


def _allocate_buffers(rank):
    src = ash.aclshmemx_malloc(
        4096,
        ash.MemType.DEVICE_SIDE,
    )
    dst = ash.aclshmemx_malloc(
        4096,
        ash.MemType.DEVICE_SIDE,
    )
    if src in (0, None):
        raise RuntimeError(
            f"PE{rank}: src malloc failed"
        )
    if dst in (0, None):
        raise RuntimeError(
            f"PE{rank}: dst malloc failed"
        )
    return src, dst


def _prepare_buffers(rank, prev, src, dst):
    pattern = bytes(
        ((i + rank * 73 + 17) % 256)
        for i in range(MSG_SIZE)
    )
    expected = bytes(
        ((i + prev * 73 + 17) % 256)
        for i in range(MSG_SIZE)
    )
    h2d(src, pattern)

    ret = acl.rt.memset(
        dst,
        4096,
        0,
        4096,
    )
    if ret != 0:
        raise RuntimeError(
            f"PE{rank}: memset failed, ret={ret}"
        )
    return expected


def _run_putmem(rank, peer, src, dst):
    stream, ret = acl.rt.create_stream()
    if ret != 0:
        raise RuntimeError(
            f"PE{rank}: create_stream failed, ret={ret}"
        )

    ash.aclshmemx_putmem_on_stream(
        dst,
        src,
        MSG_SIZE,
        peer,
        stream,
    )
    ret = acl.rt.synchronize_stream(stream)
    if ret != 0:
        raise RuntimeError(
            f"PE{rank}: stream synchronize failed, ret={ret}"
        )
    return stream


def _verify_result(rank, peer, prev, dst, expected):
    actual = d2h(
        dst,
        MSG_SIZE,
    )
    if actual != expected:
        first_bad = next(
            (
                i
                for i, (a, e)
                in enumerate(zip(actual, expected))
                if a != e
            ),
            None,
        )
        raise AssertionError(
            f"PE{rank}: bit-exact mismatch at byte "
            f"{first_bad}: "
            f"actual={actual[first_bad]}, "
            f"expected={expected[first_bad]}"
        )

    print(
        f"PE{rank}: PUTMEM_ON_STREAM_BITEXACT_PASS "
        f"bytes={MSG_SIZE} "
        f"peer={peer} "
        f"source_rank={prev}",
        flush=True,
    )


def _cleanup_resources(src, dst, stream):
    if stream is not None:
        acl.rt.destroy_stream(stream)
    if src not in (None, 0):
        ash.aclshmemx_free(
            src,
            ash.MemType.DEVICE_SIDE,
        )
    if dst not in (None, 0):
        ash.aclshmemx_free(
            dst,
            ash.MemType.DEVICE_SIDE,
        )
    core.finalize()


def _run_transfer(rank, world_size):
    peer = (rank + 1) % world_size
    prev = (rank - 1 + world_size) % world_size

    uid = broadcast_uid(rank)
    core.init(
        rank=rank,
        nranks=world_size,
        mem_size=HEAP_SIZE,
        uid=uid,
        initializer_method="uid",
    )

    src = None
    dst = None
    stream = None
    try:
        src, dst = _allocate_buffers(rank)
        expected = _prepare_buffers(
            rank, prev, src, dst
        )
        dist.barrier()

        stream = _run_putmem(
            rank, peer, src, dst
        )
        dist.barrier()

        _verify_result(
            rank, peer, prev, dst, expected
        )
        dist.barrier()
    finally:
        _cleanup_resources(
            src, dst, stream
        )


def main():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != 2:
        raise RuntimeError(
            f"This test expects 2 PEs, got {world_size}"
        )

    _run_transfer(
        rank,
        world_size,
    )


if __name__ == "__main__":
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
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
                "PUTMEM_ON_STREAM_BITEXACT_OVERALL_PASS",
                flush=True,
            )

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
