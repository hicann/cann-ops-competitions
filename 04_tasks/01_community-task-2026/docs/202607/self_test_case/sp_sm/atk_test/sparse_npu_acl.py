"""ctypes helpers for aclsparse SDDMM / SpGEMM / SpSM on NPU."""
import ctypes
import os

import numpy as np
import torch

from sparse_executor_common import resolve_sparse_lib

AclSparseStatus = ctypes.c_int

# aclsparseIndexType_t / aclsparseIndexBase_t
ACL_SPARSE_INDEX_32I = 0
ACL_SPARSE_INDEX_BASE_ZERO = 0
ACL_SPARSE_ORDER_ROW = 0

# Optional SpMat attributes (align cuSPARSE generic API when exported by libops_sparse)
ACL_SPARSE_SPMAT_FILL_MODE = 0
ACL_SPARSE_SPMAT_DIAG_TYPE = 1
ACL_SPARSE_FILL_MODE_LOWER = 0
ACL_SPARSE_FILL_MODE_UPPER = 1
ACL_SPARSE_DIAG_TYPE_NON_UNIT = 0
ACL_SPARSE_DIAG_TYPE_UNIT = 1


def _require_npu():
    import torch_npu  # noqa: F401
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available")


def load_sparse_lib():
    path = resolve_sparse_lib()
    if not path:
        raise RuntimeError("Cannot find libops_sparse.so, set OPS_SPARSE_LIB_PATH")
    return path, ctypes.CDLL(path)


def bind_common(lib, ascendcl, pointer, VoidPtr, Int64, AclDataType):
    """Bind shared aclsparse symbols used by all three ops."""
    lib.aclsparseCreate.restype = AclSparseStatus
    lib.aclsparseCreate.argtypes = [pointer(VoidPtr)]
    lib.aclsparseDestroy.restype = AclSparseStatus
    lib.aclsparseDestroy.argtypes = [VoidPtr]
    lib.aclsparseSetStream.restype = AclSparseStatus
    lib.aclsparseSetStream.argtypes = [VoidPtr, VoidPtr]

    lib.aclsparseCreateCsr.restype = AclSparseStatus
    lib.aclsparseCreateCsr.argtypes = [
        pointer(VoidPtr), Int64, Int64, Int64, VoidPtr, VoidPtr, VoidPtr,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, AclDataType,
    ]
    lib.aclsparseCreateConstCsr.restype = AclSparseStatus
    lib.aclsparseCreateConstCsr.argtypes = [
        pointer(VoidPtr), Int64, Int64, Int64, VoidPtr, VoidPtr, VoidPtr,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, AclDataType,
    ]
    lib.aclsparseDestroySpMat.restype = AclSparseStatus
    lib.aclsparseDestroySpMat.argtypes = [VoidPtr]

    lib.aclsparseCreateDnMat.restype = AclSparseStatus
    lib.aclsparseCreateDnMat.argtypes = [
        pointer(VoidPtr), Int64, Int64, Int64, VoidPtr, AclDataType, ctypes.c_int,
    ]
    lib.aclsparseCreateConstDnMat.restype = AclSparseStatus
    lib.aclsparseCreateConstDnMat.argtypes = [
        pointer(VoidPtr), Int64, Int64, Int64, VoidPtr, AclDataType, ctypes.c_int,
    ]
    lib.aclsparseDestroyDnMat.restype = AclSparseStatus
    lib.aclsparseDestroyDnMat.argtypes = [VoidPtr]

    if hasattr(lib, 'aclsparseSpMatSetAttribute'):
        lib.aclsparseSpMatSetAttribute.restype = AclSparseStatus
        lib.aclsparseSpMatSetAttribute.argtypes = [
            VoidPtr, ctypes.c_int, ctypes.c_void_p,
        ]


def bind_sddmm(lib, pointer, VoidPtr, Int64, AclDataType):
    for name in ('aclsparseSDDMMGetBufferSize', 'aclsparseSDDMMPreprocess', 'aclsparseSDDMM'):
        if not hasattr(lib, name):
            raise AttributeError(f"libops_sparse missing {name}")
    lib.aclsparseSDDMMGetBufferSize.restype = AclSparseStatus
    lib.aclsparseSDDMMGetBufferSize.argtypes = [
        VoidPtr, ctypes.c_int, ctypes.c_int, VoidPtr, VoidPtr, VoidPtr,
        VoidPtr, VoidPtr, AclDataType, ctypes.c_int, pointer(ctypes.c_size_t),
    ]
    lib.aclsparseSDDMMPreprocess.restype = AclSparseStatus
    lib.aclsparseSDDMMPreprocess.argtypes = [
        VoidPtr, ctypes.c_int, ctypes.c_int, VoidPtr, VoidPtr, VoidPtr,
        VoidPtr, VoidPtr, AclDataType, ctypes.c_int, VoidPtr,
    ]
    lib.aclsparseSDDMM.restype = AclSparseStatus
    lib.aclsparseSDDMM.argtypes = [
        VoidPtr, ctypes.c_int, ctypes.c_int, VoidPtr, VoidPtr, VoidPtr,
        VoidPtr, VoidPtr, AclDataType, ctypes.c_int, VoidPtr,
    ]


def bind_spgemm(lib, pointer, VoidPtr, Int64, AclDataType):
    required = (
        'aclsparseSpGEMMCreateDescr', 'aclsparseSpGEMMDestroyDescr',
        'aclsparseSpGEMMWorkEstimation', 'aclsparseSpGEMMCompute', 'aclsparseSpGEMMCopy',
    )
    for name in required:
        if not hasattr(lib, name):
            raise AttributeError(f"libops_sparse missing {name}")
    lib.aclsparseSpGEMMCreateDescr.restype = AclSparseStatus
    lib.aclsparseSpGEMMCreateDescr.argtypes = [pointer(VoidPtr)]
    lib.aclsparseSpGEMMDestroyDescr.restype = AclSparseStatus
    lib.aclsparseSpGEMMDestroyDescr.argtypes = [VoidPtr]
    lib.aclsparseSpGEMMWorkEstimation.restype = AclSparseStatus
    lib.aclsparseSpGEMMWorkEstimation.argtypes = [
        VoidPtr, ctypes.c_int, ctypes.c_int, VoidPtr, VoidPtr, VoidPtr,
        VoidPtr, VoidPtr, AclDataType, ctypes.c_int, VoidPtr,
        pointer(ctypes.c_size_t), VoidPtr,
    ]
    lib.aclsparseSpGEMMCompute.restype = AclSparseStatus
    lib.aclsparseSpGEMMCompute.argtypes = [
        VoidPtr, ctypes.c_int, ctypes.c_int, VoidPtr, VoidPtr, VoidPtr,
        VoidPtr, VoidPtr, AclDataType, ctypes.c_int, VoidPtr,
        pointer(ctypes.c_size_t), VoidPtr,
    ]
    lib.aclsparseSpGEMMCopy.restype = AclSparseStatus
    lib.aclsparseSpGEMMCopy.argtypes = [
        VoidPtr, ctypes.c_int, ctypes.c_int, VoidPtr, VoidPtr, VoidPtr,
        VoidPtr, VoidPtr, AclDataType, ctypes.c_int, VoidPtr,
    ]
    if hasattr(lib, 'aclsparseSpGEMMGetNumProducts'):
        lib.aclsparseSpGEMMGetNumProducts.restype = AclSparseStatus
        lib.aclsparseSpGEMMGetNumProducts.argtypes = [VoidPtr, pointer(ctypes.c_int64)]
    if hasattr(lib, 'aclsparseSpGEMMEstimateMemory'):
        lib.aclsparseSpGEMMEstimateMemory.restype = AclSparseStatus
        lib.aclsparseSpGEMMEstimateMemory.argtypes = [
            VoidPtr, ctypes.c_int, ctypes.c_int, VoidPtr, VoidPtr, VoidPtr,
            VoidPtr, VoidPtr, AclDataType, ctypes.c_int, VoidPtr,
            ctypes.c_float, pointer(ctypes.c_size_t), VoidPtr, pointer(ctypes.c_size_t),
        ]


def bind_spsm(lib, pointer, VoidPtr, Int64, AclDataType):
    required = (
        'aclsparseSpSMCreateDescr', 'aclsparseSpSMDestroyDescr',
        'aclsparseSpSMGetBufferSize', 'aclsparseSpSMAnalysis', 'aclsparseSpSMSolve',
    )
    for name in required:
        if not hasattr(lib, name):
            raise AttributeError(f"libops_sparse missing {name}")
    lib.aclsparseSpSMCreateDescr.restype = AclSparseStatus
    lib.aclsparseSpSMCreateDescr.argtypes = [pointer(VoidPtr)]
    lib.aclsparseSpSMDestroyDescr.restype = AclSparseStatus
    lib.aclsparseSpSMDestroyDescr.argtypes = [VoidPtr]
    lib.aclsparseSpSMGetBufferSize.restype = AclSparseStatus
    lib.aclsparseSpSMGetBufferSize.argtypes = [
        VoidPtr, ctypes.c_int, ctypes.c_int, VoidPtr, VoidPtr, VoidPtr,
        VoidPtr, AclDataType, ctypes.c_int, VoidPtr, pointer(ctypes.c_size_t),
    ]
    lib.aclsparseSpSMAnalysis.restype = AclSparseStatus
    lib.aclsparseSpSMAnalysis.argtypes = [
        VoidPtr, ctypes.c_int, ctypes.c_int, VoidPtr, VoidPtr, VoidPtr,
        VoidPtr, AclDataType, ctypes.c_int, VoidPtr, VoidPtr,
    ]
    lib.aclsparseSpSMSolve.restype = AclSparseStatus
    lib.aclsparseSpSMSolve.argtypes = [
        VoidPtr, ctypes.c_int, ctypes.c_int, VoidPtr, VoidPtr, VoidPtr,
        VoidPtr, AclDataType, ctypes.c_int, VoidPtr,
    ]
    if hasattr(lib, 'aclsparseSpSMUpdateMatrix'):
        lib.aclsparseSpSMUpdateMatrix.restype = AclSparseStatus
        lib.aclsparseSpSMUpdateMatrix.argtypes = [VoidPtr, VoidPtr, VoidPtr, ctypes.c_int]


def dtype_to_acl(ab_dtype, c_dtype, compute_type_id, AclDataType):
    """Map ATK dtype combo to acl value/compute types (SDDMM aware)."""
    if ab_dtype == 'fp16' and c_dtype == 'fp32':
        return AclDataType.ACL_FLOAT16, AclDataType.ACL_FLOAT, AclDataType.ACL_FLOAT
    if ab_dtype == 'fp16':
        return AclDataType.ACL_FLOAT16, AclDataType.ACL_FLOAT16, AclDataType.ACL_FLOAT
    if ab_dtype == 'bf16':
        return AclDataType.ACL_BF16, AclDataType.ACL_BF16, AclDataType.ACL_BF16
    return AclDataType.ACL_FLOAT, AclDataType.ACL_FLOAT, AclDataType.ACL_FLOAT


def elem_size(acl_dtype_enum, AclDataType):
    if acl_dtype_enum == AclDataType.ACL_FLOAT16 or acl_dtype_enum == AclDataType.ACL_BF16:
        return 2
    return 4


def check_acl_ret(ret, op_name):
    """acl 接口非 0 时立即失败，禁止 silent fallback。"""
    if ret != 0:
        raise RuntimeError(f"aclsparse {op_name} failed: ret={ret}")


def _safe(fn, *args):
    try:
        fn(*args)
    except Exception:
        pass


def set_spmat_triangular_attrs(lib, matA, fill_mode, diag_type):
    if not hasattr(lib, 'aclsparseSpMatSetAttribute'):
        return
    fill_val = ctypes.c_int(
        ACL_SPARSE_FILL_MODE_LOWER if fill_mode == 0 else ACL_SPARSE_FILL_MODE_UPPER)
    diag_val = ctypes.c_int(
        ACL_SPARSE_DIAG_TYPE_UNIT if diag_type == 1 else ACL_SPARSE_DIAG_TYPE_NON_UNIT)
    lib.aclsparseSpMatSetAttribute(
        matA, ctypes.c_int(ACL_SPARSE_SPMAT_FILL_MODE), ctypes.byref(fill_val))
    lib.aclsparseSpMatSetAttribute(
        matA, ctypes.c_int(ACL_SPARSE_SPMAT_DIAG_TYPE), ctypes.byref(diag_val))


def pad_dense_row_major(tensor, ld, rows, cols):
    if ld <= cols:
        return tensor.contiguous()
    out = torch.zeros(rows, ld, dtype=tensor.dtype, device=tensor.device)
    out[:, :cols] = tensor[:, :cols]
    return out


def scatter_csr_to_dense(m, n, row_off, col_ind, vals, out_dtype):
    dense = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for p in range(row_off[i], row_off[i + 1]):
            dense[i, col_ind[p]] = vals[p]
    return torch.from_numpy(dense).to(out_dtype)


def h2d(ascendcl, AclrtMemMallocPolicy, AclrtMemcpyKind, host_arr):
    if host_arr is None or host_arr.size == 0:
        return ctypes.c_void_p(0)
    d_ptr = ctypes.c_void_p()
    nbytes = host_arr.nbytes
    ascendcl.aclrtMalloc(ctypes.byref(d_ptr), ctypes.c_size_t(nbytes),
                         AclrtMemMallocPolicy.ACL_MEM_MALLOC_HUGE_FIRST)
    host_ptr = ctypes.c_void_p(host_arr.ctypes.data)
    ascendcl.aclrtMemcpy(d_ptr, ctypes.c_size_t(nbytes), host_ptr,
                         ctypes.c_size_t(nbytes), AclrtMemcpyKind.ACL_MEMCPY_HOST_TO_DEVICE)
    return d_ptr


def d2h(ascendcl, AclrtMemcpyKind, d_ptr, nbytes, np_dtype):
    buf = np.empty(nbytes // np.dtype(np_dtype).itemsize, dtype=np_dtype)
    ascendcl.aclrtMemcpy(ctypes.c_void_p(buf.ctypes.data), ctypes.c_size_t(nbytes),
                         d_ptr, ctypes.c_size_t(nbytes), AclrtMemcpyKind.ACL_MEMCPY_DEVICE_TO_HOST)
    return buf
