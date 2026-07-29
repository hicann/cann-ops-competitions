# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared SHMEM init / finalize fixture for multi-PE torchrun tests."""
import shmem as ash

DEFAULT_HEAP_SIZE = 64 * 1024 * 1024
DEFAULT_IP_PORT = "tcp://127.0.0.1:8766"


class ShmemSession:
    """
    Context manager that initializes ACLSHMEM with InitAttr (MTE by default)
    and finalizes on exit.
    """

    def __init__(
        self,
        pe,
        world_size,
        heap_size=DEFAULT_HEAP_SIZE,
        ip_port=DEFAULT_IP_PORT,
        engine=None,
        instance_id=0,
    ):
        self.pe = pe
        self.world_size = world_size
        self.heap_size = heap_size
        self.ip_port = ip_port
        self.engine = engine if engine is not None else ash.OpEngineType.MTE
        self.instance_id = instance_id
        self._inited = False

    def __enter__(self):
        ret = ash.set_conf_store_tls(False, "")
        if ret != 0:
            raise RuntimeError(f"set_conf_store_tls failed, ret={ret}")

        attributes = ash.InitAttr()
        attributes.my_rank = self.pe
        attributes.n_ranks = self.world_size
        attributes.local_mem_size = self.heap_size
        attributes.ip_port = self.ip_port
        attributes.option_attr.data_op_engine_type = self.engine
        if hasattr(attributes, "instance_id"):
            attributes.instance_id = self.instance_id

        ret = ash.aclshmem_init(attributes)
        if ret != 0:
            raise RuntimeError(f"aclshmem_init failed, ret={ret}")
        self._inited = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._inited:
            try:
                ash.aclshmem_finalize(self.instance_id)
            except TypeError:
                ash.aclshmem_finalize()
        return False
