import ctypes
import os
import sys

import numpy as np
import torch

_ATK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ATK_DIR not in sys.path:
    sys.path.insert(0, _ATK_DIR)

from sparse_executor_common import (
    csr_to_dense,
    generate_triangular_csr,
    require_cuda,
    seed_from_dims,
    to_result_tensor,
)
from sparse_gpu_cusparse import cusparse_spsm
from sparse_npu_acl import (
    _require_npu,
    _safe,
    bind_common,
    bind_spsm,
    check_acl_ret,
    h2d,
    load_sparse_lib,
    set_spmat_triangular_attrs,
)

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

logging = Logger().get_logger()


def _triangular_solve_numpy(A, B, lower=True, unit_diag=False, acc=np.float64):
    A = A.astype(acc, copy=False)
    m = A.shape[0]
    nrhs = B.shape[1]
    X = np.zeros((m, nrhs), dtype=acc)
    rhs = B.astype(acc, copy=False)
    if lower:
        for i in range(m):
            diag = acc(1.0) if unit_diag else A[i, i]
            X[i, :] = (rhs[i, :] - A[i, :i] @ X[:i, :]) / diag
    else:
        for i in range(m - 1, -1, -1):
            diag = acc(1.0) if unit_diag else A[i, i]
            X[i, :] = (rhs[i, :] - A[i, i + 1:] @ X[i + 1:, :]) / diag
    return X


@register("execute_aclSparseSpsm")
class AclSparseSpsm(BaseApi):
    """SpSM: op(A) X = alpha * op(B); A sparse triangular, B/C dense [m, nrhs]."""

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.denseB = None
        self.denseC = None
        self.avg_degree = 10.0
        self.alpha_val = 1.0
        self.opA_val = 0
        self.opB_val = 0
        self.fill_mode = 0
        self.diag_type = 0
        self.nrhs = 1
        self.in_place = 0
        self.null_values = 0
        self.update_matrix = 0
        self.alg_val = 0
        self.computeType_val = 0
        self.out_dtype = torch.float32
        self.m = 0
        self.nnz_a = 0
        self.a_row_off = self.a_col_ind = self.a_vals = None

    def init_by_input_data(self, input_data: InputDataset):
        self.denseB = input_data.kwargs['denseB'].clone()
        self.denseC = input_data.kwargs['denseC'].clone()
        self.avg_degree = float(input_data.kwargs['avgDegree'])
        self.alpha_val = float(input_data.kwargs['alpha'])
        self.opA_val = int(input_data.kwargs['opA'])
        self.opB_val = int(input_data.kwargs['opB'])
        self.fill_mode = int(input_data.kwargs['fillMode'])
        self.diag_type = int(input_data.kwargs['diagType'])
        self.nrhs = int(input_data.kwargs['nrhs'])
        self.in_place = int(input_data.kwargs['inPlace'])
        self.null_values = int(input_data.kwargs['nullValues'])
        self.update_matrix = int(input_data.kwargs['updateMatrix'])
        self.alg_val = int(input_data.kwargs['alg'])
        self.computeType_val = int(input_data.kwargs['computeType'])
        self.out_dtype = input_data.kwargs['out'].dtype
        self.m = self.denseB.shape[0]

        near_singular = int(input_data.kwargs.get('scenario', 0)) == 7
        seed, rng, np_rng = seed_from_dims('SpSM', self.m, int(self.avg_degree * 100), self.nrhs)
        self.a_row_off, self.a_col_ind, self.a_vals = generate_triangular_csr(
            self.m, self.avg_degree, self.fill_mode, self.diag_type,
            dtype='fp32', rng=rng, np_rng=np_rng, near_singular=near_singular)
        self.nnz_a = len(self.a_vals)

    def _apply_op_b(self, B):
        if self.opB_val == 0:
            return B
        return B.T.contiguous()

    def _reference(self, high_precision=False):
        m, nrhs = self.m, self.nrhs
        acc = np.float64 if high_precision else np.float32
        a_vals = self.a_vals
        if self.update_matrix:
            # Align with GPU/NPU: cusparseSpSM_updateMatrix + second solve (RandomState(0) perturb).
            a_vals = self.a_vals.copy().astype(np.float64)
            a_vals = a_vals + np.random.RandomState(0).uniform(-0.1, 0.1, size=a_vals.shape)
            a_vals = a_vals.astype(np.float32)

        A_dense = csr_to_dense(self.a_row_off, self.a_col_ind, a_vals, (m, m), acc_dtype=acc)
        A_op = A_dense.T if self.opA_val == 1 else A_dense
        is_lower = (self.fill_mode == 0 and self.opA_val == 0) or (self.fill_mode == 1 and self.opA_val == 1)
        B_np = self._apply_op_b(self.denseB[:m, :nrhs]).numpy().astype(acc, copy=False)
        X = _triangular_solve_numpy(
            A_op, self.alpha_val * B_np, lower=is_lower,
            unit_diag=(self.diag_type == 1), acc=acc)
        if high_precision:
            return to_result_tensor(X, self.out_dtype, high_precision=True)
        return torch.from_numpy(X.astype(np.float32)).to(self.out_dtype)

    def _cpu_call(self):
        return self._reference(high_precision=True)

    def _npu_call(self):
        import torch_npu  # noqa: F401
        from atk.tasks.backends.lib_interface.acl_wrapper import (
            AclDataType, AclrtMemMallocPolicy, AclrtMemcpyKind, Int64, VoidPtr, ascendcl, pointer,
        )

        _require_npu()
        _, lib = load_sparse_lib()
        bind_common(lib, ascendcl, pointer, VoidPtr, Int64, AclDataType)
        bind_spsm(lib, pointer, VoidPtr, Int64, AclDataType)

        m, nrhs = self.m, self.nrhs
        acl_type = AclDataType.ACL_FLOAT

        B_in = self._apply_op_b(self.denseB[:m, :nrhs]).contiguous().npu()
        if self.in_place:
            C_out = B_in
        else:
            C_out = self.denseC[:m, :nrhs].contiguous().npu()
        torch.npu.synchronize()

        handle = VoidPtr()
        spsm_descr = VoidPtr()
        matA = VoidPtr()
        matB = VoidPtr()
        matC = VoidPtr()
        dRow = dCol = dVals = VoidPtr()
        dBuffer = VoidPtr()
        stream = VoidPtr()

        try:
            ascendcl.aclrtCreateStream(ctypes.byref(stream))
            lib.aclsparseCreate(ctypes.byref(handle))
            lib.aclsparseSetStream(handle, stream)
            lib.aclsparseSpSMCreateDescr(ctypes.byref(spsm_descr))

            dRow = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, self.a_row_off)
            dCol = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, self.a_col_ind)
            dVals = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, self.a_vals)

            lib.aclsparseCreateConstCsr(
                ctypes.byref(matA), Int64(m), Int64(m), Int64(self.nnz_a),
                dRow, dCol, dVals, ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0), acl_type)
            set_spmat_triangular_attrs(lib, matA, self.fill_mode, self.diag_type)

            dB_ptr = ctypes.c_void_p(0 if self.null_values else B_in.data_ptr())
            dC_ptr = ctypes.c_void_p(0 if self.null_values else C_out.data_ptr())

            lib.aclsparseCreateConstDnMat(
                ctypes.byref(matB), Int64(m), Int64(nrhs), Int64(nrhs),
                dB_ptr, acl_type, ctypes.c_int(0))
            lib.aclsparseCreateDnMat(
                ctypes.byref(matC), Int64(m), Int64(nrhs), Int64(nrhs),
                dC_ptr, acl_type, ctypes.c_int(0))

            _alpha = ctypes.c_float(self.alpha_val)
            alpha = ctypes.c_void_p(ctypes.addressof(_alpha))
            opA = ctypes.c_int(self.opA_val)
            opB = ctypes.c_int(self.opB_val)
            alg = ctypes.c_int(self.alg_val)

            buf_size = ctypes.c_size_t(0)
            ret = lib.aclsparseSpSMGetBufferSize(
                handle, opA, opB, alpha, matA, matB, matC,
                acl_type, alg, spsm_descr, ctypes.byref(buf_size))
            check_acl_ret(ret, "SpSMGetBufferSize")
            if buf_size.value > 0:
                ascendcl.aclrtMalloc(ctypes.byref(dBuffer), buf_size,
                                     AclrtMemMallocPolicy.ACL_MEM_MALLOC_HUGE_FIRST)

            ret = lib.aclsparseSpSMAnalysis(
                handle, opA, opB, alpha, matA, matB, matC,
                acl_type, alg, spsm_descr, dBuffer)
            check_acl_ret(ret, "SpSMAnalysis")

            if self.null_values:
                lib.aclsparseDestroyDnMat(matB)
                lib.aclsparseDestroyDnMat(matC)
                dB_ptr = ctypes.c_void_p(B_in.data_ptr())
                dC_ptr = ctypes.c_void_p(C_out.data_ptr())
                lib.aclsparseCreateConstDnMat(
                    ctypes.byref(matB), Int64(m), Int64(nrhs), Int64(nrhs),
                    dB_ptr, acl_type, ctypes.c_int(0))
                lib.aclsparseCreateDnMat(
                    ctypes.byref(matC), Int64(m), Int64(nrhs), Int64(nrhs),
                    dC_ptr, acl_type, ctypes.c_int(0))

            ret = lib.aclsparseSpSMSolve(
                handle, opA, opB, alpha, matA, matB, matC,
                acl_type, alg, spsm_descr)
            check_acl_ret(ret, "SpSMSolve")

            if self.update_matrix:
                if not hasattr(lib, 'aclsparseSpSMUpdateMatrix'):
                    raise RuntimeError("aclsparseSpSMUpdateMatrix not found in libops_sparse.so")
                new_vals = self.a_vals.copy()
                np_rng = np.random.RandomState(0)
                new_vals = new_vals + np_rng.uniform(-0.1, 0.1, size=new_vals.shape).astype(np.float32)
                dNewVals = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, new_vals)
                try:
                    lib.aclsparseSpSMUpdateMatrix(handle, spsm_descr, dNewVals, ctypes.c_int(0))
                    ret = lib.aclsparseSpSMSolve(
                        handle, opA, opB, alpha, matA, matB, matC,
                        acl_type, alg, spsm_descr)
                    check_acl_ret(ret, "SpSMSolve(after UpdateMatrix)")
                finally:
                    _safe(ascendcl.aclrtFree, dNewVals)

            ascendcl.aclrtSynchronizeStream(stream)

            result_bytes = 4 * m * nrhs
            result_buf = (ctypes.c_uint8 * result_bytes)()
            ascendcl.aclrtMemcpy(ctypes.c_void_p(ctypes.addressof(result_buf)),
                                 ctypes.c_size_t(result_bytes),
                                 ctypes.c_void_p(C_out.data_ptr()),
                                 ctypes.c_size_t(result_bytes),
                                 AclrtMemcpyKind.ACL_MEMCPY_DEVICE_TO_HOST)
            arr = np.frombuffer(result_buf, dtype=np.float32).copy().reshape(m, nrhs)
            return torch.from_numpy(arr).to(self.out_dtype)
        finally:
            _safe(lib.aclsparseSpSMDestroyDescr, spsm_descr)
            _safe(lib.aclsparseDestroySpMat, matA)
            _safe(lib.aclsparseDestroyDnMat, matB)
            _safe(lib.aclsparseDestroyDnMat, matC)
            _safe(lib.aclsparseDestroy, handle)
            _safe(ascendcl.aclrtFree, dBuffer)
            _safe(ascendcl.aclrtFree, dRow)
            _safe(ascendcl.aclrtFree, dCol)
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
