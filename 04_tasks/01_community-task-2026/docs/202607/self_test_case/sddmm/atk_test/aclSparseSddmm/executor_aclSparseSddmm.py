import ctypes
import os
import sys

import numpy as np
import torch

_ATK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ATK_DIR not in sys.path:
    sys.path.insert(0, _ATK_DIR)

from sparse_executor_common import (
    apply_sddmm_mask,
    generate_csr_data,
    infer_value_dtype,
    logical_dense_from_tensor,
    nnz_from_sparsity,
    require_cuda,
    seed_from_dims,
    to_result_tensor,
)
from sparse_gpu_cusparse import cusparse_sddmm
from sparse_npu_acl import (
    _require_npu,
    _safe,
    bind_common,
    bind_sddmm,
    check_acl_ret,
    d2h,
    dtype_to_acl,
    elem_size,
    h2d,
    load_sparse_lib,
    pad_dense_row_major,
    scatter_csr_to_dense,
)

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

logging = Logger().get_logger()


@register("execute_aclSparseSddmm")
class AclSparseSddmm(BaseApi):
    """SDDMM: C = alpha * (op(A)*op(B)) o spy(C) + beta * C"""

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.denseA = None
        self.denseB = None
        self.denseC = None
        self.sparsity = None
        self.alpha_val = 0.0
        self.beta_val = 0.0
        self.opA_val = 0
        self.opB_val = 0
        self.orderA_val = 0
        self.orderB_val = 0
        self.ldA_val = 0
        self.ldB_val = 0
        self.computeType_val = 0
        self.alg_val = 0
        self.out_dtype = torch.float32
        self.ab_dtype_name = 'fp32'
        self.c_dtype_name = 'fp32'
        self.m = self.n = self.k = 0
        self.nnz_c = 0
        self.c_row_off = None
        self.c_col_ind = None
        self.c_vals_host = None

    def init_by_input_data(self, input_data: InputDataset):
        self.denseA = input_data.kwargs['denseA'].clone()
        self.denseB = input_data.kwargs['denseB'].clone()
        self.denseC = input_data.kwargs['denseC'].clone()
        self.sparsity = float(input_data.kwargs['sparsity'])
        self.alpha_val = float(input_data.kwargs['alpha'])
        self.beta_val = float(input_data.kwargs['beta'])
        self.opA_val = int(input_data.kwargs['opA'])
        self.opB_val = int(input_data.kwargs['opB'])
        self.orderA_val = int(input_data.kwargs['orderA'])
        self.orderB_val = int(input_data.kwargs['orderB'])
        self.ldA_val = int(input_data.kwargs['ldA'])
        self.ldB_val = int(input_data.kwargs['ldB'])
        self.computeType_val = int(input_data.kwargs['computeType'])
        self.alg_val = int(input_data.kwargs['alg'])
        self.out_dtype = input_data.kwargs['out'].dtype

        self.ab_dtype_name = infer_value_dtype(self.denseA)
        self.c_dtype_name = infer_value_dtype(self.denseC)
        self.m = self.denseC.shape[0]
        self.n = self.denseC.shape[1]
        self.k = self.denseB.shape[0] if self.opB_val == 0 else self.denseB.shape[1]

        self.nnz_c = nnz_from_sparsity(self.m, self.n, self.sparsity)
        seed, rng, np_rng = seed_from_dims(self.m, self.n, self.nnz_c)
        self.c_row_off, self.c_col_ind, _ = generate_csr_data(
            self.m, self.n, max(self.nnz_c, 0), self.c_dtype_name, rng=rng, np_rng=np_rng)

        self.c_vals_host = np.zeros(max(self.nnz_c, 0), dtype=np.float32)
        for i in range(self.m):
            for p in range(self.c_row_off[i], self.c_row_off[i + 1]):
                self.c_vals_host[p] = float(self.denseC[i, self.c_col_ind[p]].item())

    def _reference(self, high_precision=False):
        m, n, k = self.m, self.n, self.k
        A = logical_dense_from_tensor(self.denseA, self.opA_val, m, k)
        B = logical_dense_from_tensor(self.denseB, self.opB_val, k, n)
        C0 = self.denseC[:m, :n]

        uniform_fp32 = self.ab_dtype_name == 'fp32' and self.c_dtype_name == 'fp32'
        if high_precision and uniform_fp32:
            acc = torch.float64
            full = self.alpha_val * A.to(acc) @ B.to(acc) + self.beta_val * C0.to(acc)
            result = apply_sddmm_mask(full.cpu().numpy(), self.c_row_off, self.c_col_ind)
            return to_result_tensor(result, self.out_dtype, high_precision=True)

        if self.ab_dtype_name == 'fp16':
            A_t = A.to(torch.float16).to(torch.float32)
            B_t = B.to(torch.float16).to(torch.float32)
        else:
            A_t = A.to(torch.float32)
            B_t = B.to(torch.float32)
        C_t = C0.to(torch.float16 if self.c_dtype_name == 'fp16' else torch.float32)
        if self.c_dtype_name == 'fp16':
            C_t = C_t.to(torch.float16).to(torch.float32)

        full = self.alpha_val * A_t @ B_t + self.beta_val * C_t
        result_np = apply_sddmm_mask(full.cpu().numpy(), self.c_row_off, self.c_col_ind)
        if self.c_dtype_name == 'fp16' or self.out_dtype == torch.float16:
            result_np = result_np.astype(np.float32).astype(np.float16).astype(np.float32)
        return torch.from_numpy(result_np).to(self.out_dtype)

    def _cpu_call(self):
        return self._reference(high_precision=True)

    def _npu_call(self):
        import torch_npu  # noqa: F401
        from atk.tasks.backends.lib_interface.acl_wrapper import (
            AclDataType, AclrtMemMallocPolicy, AclrtMemcpyKind, Int64, VoidPtr, ascendcl, pointer,
        )

        _require_npu()
        m, n, k, nnz = self.m, self.n, self.k, self.nnz_c
        if nnz == 0:
            return torch.zeros(m, n, dtype=self.out_dtype)

        _, lib = load_sparse_lib()
        bind_common(lib, ascendcl, pointer, VoidPtr, Int64, AclDataType)
        bind_sddmm(lib, pointer, VoidPtr, Int64, AclDataType)

        acl_ab, acl_c, acl_compute = dtype_to_acl(
            self.ab_dtype_name, self.c_dtype_name, self.computeType_val, AclDataType)
        val_esize = elem_size(acl_c, AclDataType)

        a_rows, a_cols = (m, k) if self.opA_val == 0 else (k, m)
        b_rows, b_cols = (k, n) if self.opB_val == 0 else (n, k)
        ld_a = self.ldA_val if self.ldA_val > 0 else (k if self.opA_val == 0 else m)
        ld_b = self.ldB_val if self.ldB_val > 0 else (n if self.opB_val == 0 else k)

        dA = pad_dense_row_major(self.denseA.npu(), ld_a, a_rows, a_cols)
        dB = pad_dense_row_major(self.denseB.npu(), ld_b, b_rows, b_cols)
        torch.npu.synchronize()

        c_vals_np = self.c_vals_host.astype(
            np.float16 if self.c_dtype_name == 'fp16' else np.float32, copy=False)

        handle = VoidPtr()
        matA = VoidPtr()
        matB = VoidPtr()
        matC = VoidPtr()
        dRowOff = VoidPtr()
        dColInd = VoidPtr()
        dVals = VoidPtr()
        dBuffer = VoidPtr()
        stream = VoidPtr()

        try:
            ascendcl.aclrtCreateStream(ctypes.byref(stream))
            lib.aclsparseCreate(ctypes.byref(handle))
            lib.aclsparseSetStream(handle, stream)

            dRowOff = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, self.c_row_off)
            dColInd = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, self.c_col_ind)
            dVals = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, c_vals_np)

            lib.aclsparseCreateCsr(
                ctypes.byref(matC), Int64(m), Int64(n), Int64(nnz),
                dRowOff, dColInd, dVals,
                ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0), acl_c)

            lib.aclsparseCreateConstDnMat(
                ctypes.byref(matA), Int64(m), Int64(k), Int64(ld_a),
                ctypes.c_void_p(dA.data_ptr()), acl_ab, ctypes.c_int(self.orderA_val))
            lib.aclsparseCreateConstDnMat(
                ctypes.byref(matB), Int64(k), Int64(n), Int64(ld_b),
                ctypes.c_void_p(dB.data_ptr()), acl_ab, ctypes.c_int(self.orderB_val))

            _alpha = ctypes.c_float(self.alpha_val)
            _beta = ctypes.c_float(self.beta_val)
            alpha = ctypes.c_void_p(ctypes.addressof(_alpha))
            beta = ctypes.c_void_p(ctypes.addressof(_beta))
            opA = ctypes.c_int(self.opA_val)
            opB = ctypes.c_int(self.opB_val)
            alg = ctypes.c_int(self.alg_val)

            buffer_size = ctypes.c_size_t(0)
            ret = lib.aclsparseSDDMMGetBufferSize(
                handle, opA, opB, alpha, matA, matB, beta, matC,
                acl_compute, alg, ctypes.byref(buffer_size))
            check_acl_ret(ret, "SDDMMGetBufferSize")

            if buffer_size.value > 0:
                ascendcl.aclrtMalloc(ctypes.byref(dBuffer), buffer_size,
                                     AclrtMemMallocPolicy.ACL_MEM_MALLOC_HUGE_FIRST)

            ret = lib.aclsparseSDDMMPreprocess(
                handle, opA, opB, alpha, matA, matB, beta, matC,
                acl_compute, alg, dBuffer)
            check_acl_ret(ret, "SDDMMPreprocess")

            ret = lib.aclsparseSDDMM(
                handle, opA, opB, alpha, matA, matB, beta, matC,
                acl_compute, alg, dBuffer)
            check_acl_ret(ret, "SDDMM")

            ascendcl.aclrtSynchronizeStream(stream)

            out_vals = d2h(ascendcl, AclrtMemcpyKind, dVals, val_esize * max(nnz, 1),
                           np.float16 if self.c_dtype_name == 'fp16' else np.float32)
            if nnz == 0:
                return torch.zeros(m, n, dtype=self.out_dtype)
            return scatter_csr_to_dense(m, n, self.c_row_off, self.c_col_ind,
                                      out_vals[:nnz], self.out_dtype)
        finally:
            _safe(lib.aclsparseDestroySpMat, matC)
            _safe(lib.aclsparseDestroyDnMat, matA)
            _safe(lib.aclsparseDestroyDnMat, matB)
            _safe(lib.aclsparseDestroy, handle)
            _safe(ascendcl.aclrtFree, dBuffer)
            _safe(ascendcl.aclrtFree, dRowOff)
            _safe(ascendcl.aclrtFree, dColInd)
            _safe(ascendcl.aclrtFree, dVals)
            _safe(ascendcl.aclrtDestroyStream, stream)
            try:
                torch.npu.empty_cache()
            except Exception:
                pass

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            return self._cpu_call()
        if self.device == "npu":
            return self._npu_call()
        raise RuntimeError(f"unsupported backend: {self.device}")
