import sys

TEST_ROOT = (
    "/workspace/cann-ops-competitions/04_tasks/01_community-task-2026/"
    "docs/202607/self_test_case/shmem_python/tests"
)
sys.path.insert(0, TEST_ROOT)

from common.env import bootstrap_torch_dist
from common.shmem_fixture import ShmemSession

import shmem as ash

MSG_SIZE = 4096


def main():
    pe, world_size = bootstrap_torch_dist()

    import torch
    import torch_npu  # noqa: F401

    print(
        f"pe[{pe}] world_size={world_size} "
        f"engine={ash.OpEngineType.SDMA}"
    )

    with ShmemSession(
        pe,
        world_size,
        engine=ash.OpEngineType.SDMA,
    ):
        print(f"pe[{pe}] SDMA_INIT_OK")

        ptr = ash.aclshmem_malloc(MSG_SIZE)
        assert ptr not in (0, None)

        peer = (pe + 1) % world_size

        stream = torch.npu.Stream()
        stream_ptr = stream.npu_stream

        with torch.npu.stream(stream):
            ash.aclshmemx_putmem_on_stream(
                ptr,
                ptr,
                MSG_SIZE,
                peer,
                stream_ptr,
            )

        stream.synchronize()

        print(f"pe[{pe}] SDMA_PUT_STREAM_OK")

        ash.aclshmem_free(ptr)

    print(f"pe[{pe}] SDMA_FINALIZE_OK")


if __name__ == "__main__":
    main()
