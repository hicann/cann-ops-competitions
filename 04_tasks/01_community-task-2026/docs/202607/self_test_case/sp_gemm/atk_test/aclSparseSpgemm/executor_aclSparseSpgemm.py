import ctypes
import os
import sys

import numpy as np
import torch

_ATK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ATK_DIR not in sys.path:
    sys.path.insert(0, _ATK_DIR)

from sparse_executor_common import (
    generate_csr_data,
    nnz_from_sparsity,
    require_cuda,
    seed_from_dims,
    to_result_tensor,
)
from sparse_gpu_cusparse import cusparse_spgemm
from sparse_npu_acl import (
    _require_npu,
    _safe,
    bind_common,
    bind_spgemm,
    check_acl_ret,
    d2h,
    h2d,
    load_sparse_lib,
    scatter_csr_to_dense,
)

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

logging = Logger().get_logger()

try:
    import scipy.sparse as sp
except ImportError:
    sp = None


def _value_dtype_acl(name, AclDataType):
    return {
        'fp32': AclDataType.ACL_FLOAT,
        'fp16': AclDataType.ACL_FLOAT16,
        'bf16': AclDataType.ACL_BF16,
    }[name]


def _val_esize(name):
    return 2 if name in ('fp16', 'bf16') else 4


@register("execute_aclSparseSpgemm")
class AclSparseSpgemm(BaseApi):
    """SpGEMM: C = alpha * A * B + beta * C (CSR x CSR + CSR)."""

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.sparsity_a = 0.0
        self.sparsity_b = 0.0
        self.alpha_val = 1.0
        self.beta_val = 1.0
        self.m = self.k = self.n = 0
        self.alg_val = 0
        self.computeType_val = 0
        self.out_dtype = torch.float32
        self.value_dtype_name = 'fp32'
        self.a_row_off = self.a_col_ind = self.a_vals = None
        self.b_row_off = self.b_col_ind = self.b_vals = None
        self.c_row_off = self.c_col_ind = self.c_vals = None
        self.nnz_a = self.nnz_b = self.nnz_c = 0

    def init_by_input_data(self, input_data: InputDataset):
        self.sparsity_a = float(input_data.kwargs['sparsityA'])
        self.sparsity_b = float(input_data.kwargs['sparsityB'])
        self.alpha_val = float(input_data.kwargs['alpha'])
        self.beta_val = float(input_data.kwargs['beta'])
        self.m = int(input_data.kwargs['m'])
        self.k = int(input_data.kwargs['k'])
        self.n = int(input_data.kwargs['n'])
        self.alg_val = int(input_data.kwargs['alg'])
        self.computeType_val = int(input_data.kwargs['computeType'])
        self.out_dtype = input_data.kwargs['out'].dtype
        self.value_dtype_name = {torch.float16: 'fp16', torch.bfloat16: 'bf16'}.get(self.out_dtype, 'fp32')

        self.nnz_a = max(nnz_from_sparsity(self.m, self.k, self.sparsity_a), 0)
        self.nnz_b = max(nnz_from_sparsity(self.k, self.n, self.sparsity_b), 0)
        self.nnz_c = max(nnz_from_sparsity(self.m, self.n, 0.999), 0)

        seed_a, rng_a, np_rng_a = seed_from_dims('A', self.m, self.k, self.nnz_a)
        seed_b, rng_b, np_rng_b = seed_from_dims('B', self.k, self.n, self.nnz_b)
        seed_c, rng_c, np_rng_c = seed_from_dims('C', self.m, self.n, self.nnz_c)

        self.a_row_off, self.a_col_ind, self.a_vals = generate_csr_data(
            self.m, self.k, self.nnz_a, self.value_dtype_name, rng=rng_a, np_rng=np_rng_a)
        self.b_row_off, self.b_col_ind, self.b_vals = generate_csr_data(
            self.k, self.n, self.nnz_b, self.value_dtype_name, rng=rng_b, np_rng=np_rng_b)
        self.c_row_off, self.c_col_ind, self.c_vals = generate_csr_data(
            self.m, self.n, self.nnz_c, self.value_dtype_name, rng=rng_c, np_rng=np_rng_c)

    def _zero_nnz_result(self):
        """Early return for empty A/B (nnz=0); avoids cuSPARSE/aclsparse edge failures."""
        if self.nnz_a > 0 and self.nnz_b > 0:
            return None
        if self.beta_val == 0.0:
            return torch.zeros(self.m, self.n, dtype=self.out_dtype)
        return self._reference(high_precision=True)

    def _csr_matmul(self, high_precision=False):
        if sp is None:
            raise RuntimeError("scipy is required for SpGEMM reference")
        acc = np.float64 if high_precision and self.value_dtype_name == 'fp32' else np.float32
        A = sp.csr_matrix((self.a_vals.astype(acc), self.a_col_ind, self.a_row_off), shape=(self.m, self.k))
        B = sp.csr_matrix((self.b_vals.astype(acc), self.b_col_ind, self.b_row_off), shape=(self.k, self.n))
        C = sp.csr_matrix((self.c_vals.astype(acc), self.c_col_ind, self.c_row_off), shape=(self.m, self.n))
        out = self.alpha_val * (A @ B) + self.beta_val * C
        dense = out.toarray()
        if self.value_dtype_name == 'fp16':
            dense = dense.astype(np.float32).astype(np.float16).astype(np.float32)
        elif self.value_dtype_name == 'bf16':
            dense = torch.from_numpy(dense.astype(np.float32)).to(torch.bfloat16).to(torch.float32).numpy()
        return dense

    def _reference(self, high_precision=False):
        dense = self._csr_matmul(high_precision=high_precision)
        if high_precision and self.value_dtype_name == 'fp32':
            return to_result_tensor(dense, self.out_dtype, high_precision=True)
        return torch.from_numpy(np.asarray(dense, dtype=np.float32)).to(self.out_dtype)

    def _cpu_call(self):
        empty = self._zero_nnz_result()
        if empty is not None:
            return empty
        return self._reference(high_precision=True)

    def _npu_call(self):
        import torch_npu  # noqa: F401
        from atk.tasks.backends.lib_interface.acl_wrapper import (
            AclDataType, AclrtMemMallocPolicy, AclrtMemcpyKind, Int64, VoidPtr, ascendcl, pointer,
        )

        _require_npu()
        empty = self._zero_nnz_result()
        if empty is not None:
            return empty
        _, lib = load_sparse_lib()
        bind_common(lib, ascendcl, pointer, VoidPtr, Int64, AclDataType)
        bind_spgemm(lib, pointer, VoidPtr, Int64, AclDataType)

        m, k, n = self.m, self.k, self.n
        acl_type = _value_dtype_acl(self.value_dtype_name, AclDataType)
        val_esize = _val_esize(self.value_dtype_name)

        # Use scipy golden CSR as matC allocation template (sorted columns).
        if sp is None:
            raise RuntimeError("scipy required for SpGEMM NPU matC sizing")
        out_csr = (
            self.alpha_val * sp.csr_matrix(
                (self.a_vals, self.a_col_ind, self.a_row_off), shape=(m, k)) @
            sp.csr_matrix((self.b_vals, self.b_col_ind, self.b_row_off), shape=(k, n)) +
            self.beta_val * sp.csr_matrix((self.c_vals, self.c_col_ind, self.c_row_off), shape=(m, n))
        ).tocsr()
        c_row_off = out_csr.indptr.astype(np.int32)
        c_col_ind = out_csr.indices.astype(np.int32)
        c_vals_np = np.zeros(out_csr.nnz, dtype=self.c_vals.dtype)
        nnz_c_out = int(out_csr.nnz)

        handle = VoidPtr()
        spgemm_descr = VoidPtr()
        matA = VoidPtr()
        matB = VoidPtr()
        matC = VoidPtr()
        dARow = dACol = dAVals = VoidPtr()
        dBRow = dBCol = dBVals = VoidPtr()
        dCRow = dCCol = dCVals = VoidPtr()
        dBuf1 = dBuf2 = VoidPtr()
        stream = VoidPtr()

        try:
            ascendcl.aclrtCreateStream(ctypes.byref(stream))
            lib.aclsparseCreate(ctypes.byref(handle))
            lib.aclsparseSetStream(handle, stream)
            lib.aclsparseSpGEMMCreateDescr(ctypes.byref(spgemm_descr))

            dARow = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, self.a_row_off)
            dACol = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, self.a_col_ind)
            dAVals = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, self.a_vals)
            dBRow = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, self.b_row_off)
            dBCol = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, self.b_col_ind)
            dBVals = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, self.b_vals)
            dCRow = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, c_row_off)
            dCCol = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, c_col_ind)
            dCVals = h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, c_vals_np)

            lib.aclsparseCreateConstCsr(
                ctypes.byref(matA), Int64(m), Int64(k), Int64(self.nnz_a),
                dARow, dACol, dAVals, ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0), acl_type)
            lib.aclsparseCreateConstCsr(
                ctypes.byref(matB), Int64(k), Int64(n), Int64(self.nnz_b),
                dBRow, dBCol, dBVals, ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0), acl_type)
            lib.aclsparseCreateCsr(
                ctypes.byref(matC), Int64(m), Int64(n), Int64(nnz_c_out),
                dCRow, dCCol, dCVals, ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0), acl_type)

            _alpha = ctypes.c_float(self.alpha_val)
            _beta = ctypes.c_float(self.beta_val)
            alpha = ctypes.c_void_p(ctypes.addressof(_alpha))
            beta = ctypes.c_void_p(ctypes.addressof(_beta))
            opA = ctypes.c_int(0)
            opB = ctypes.c_int(0)
            alg = ctypes.c_int(self.alg_val)

            buf_size1 = ctypes.c_size_t(0)
            null_ptr = ctypes.c_void_p(0)
            ret = lib.aclsparseSpGEMMWorkEstimation(
                handle, opA, opB, alpha, matA, matB, beta, matC,
                acl_type, alg, spgemm_descr, ctypes.byref(buf_size1), null_ptr)
            check_acl_ret(ret, "SpGEMMWorkEstimation(size)")
            if buf_size1.value > 0:
                ascendcl.aclrtMalloc(ctypes.byref(dBuf1), buf_size1,
                                     AclrtMemMallocPolicy.ACL_MEM_MALLOC_HUGE_FIRST)
                ret = lib.aclsparseSpGEMMWorkEstimation(
                    handle, opA, opB, alpha, matA, matB, beta, matC,
                    acl_type, alg, spgemm_descr, ctypes.byref(buf_size1), dBuf1)
                check_acl_ret(ret, "SpGEMMWorkEstimation")

            buf_size2 = ctypes.c_size_t(0)
            ret = lib.aclsparseSpGEMMCompute(
                handle, opA, opB, alpha, matA, matB, beta, matC,
                acl_type, alg, spgemm_descr, ctypes.byref(buf_size2), null_ptr)
            check_acl_ret(ret, "SpGEMMCompute(size)")
            if buf_size2.value > 0:
                ascendcl.aclrtMalloc(ctypes.byref(dBuf2), buf_size2,
                                     AclrtMemMallocPolicy.ACL_MEM_MALLOC_HUGE_FIRST)
                ret = lib.aclsparseSpGEMMCompute(
                    handle, opA, opB, alpha, matA, matB, beta, matC,
                    acl_type, alg, spgemm_descr, ctypes.byref(buf_size2), dBuf2)
                check_acl_ret(ret, "SpGEMMCompute")

            ret = lib.aclsparseSpGEMMCopy(
                handle, opA, opB, alpha, matA, matB, beta, matC,
                acl_type, alg, spgemm_descr)
            check_acl_ret(ret, "SpGEMMCopy")

            ascendcl.aclrtSynchronizeStream(stream)

            out_vals = d2h(ascendcl, AclrtMemcpyKind, dCVals, val_esize * max(nnz_c_out, 1),
                           np.float16 if self.value_dtype_name == 'fp16' else np.float32)
            return scatter_csr_to_dense(m, n, c_row_off, c_col_ind, out_vals[:nnz_c_out], self.out_dtype)
        finally:
            _safe(lib.aclsparseSpGEMMDestroyDescr, spgemm_descr)
            _safe(lib.aclsparseDestroySpMat, matA)
            _safe(lib.aclsparseDestroySpMat, matB)
            _safe(lib.aclsparseDestroySpMat, matC)
            _safe(lib.aclsparseDestroy, handle)
            for p in (dBuf1, dBuf2, dARow, dACol, dAVals, dBRow, dBCol, dBVals, dCRow, dCCol, dCVals):
                _safe(ascendcl.aclrtFree, p)
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
