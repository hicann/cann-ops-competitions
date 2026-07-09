"""ctypes cuSPARSE GPU helpers for SDDMM / SpGEMM / SpSM ATK executors.

All three ops call libcusparse Generic API directly (no cupy/dgl).
SpGEMM flow: NVIDIA CUDALibrarySamples cuSPARSE/spgemm.
"""
import ctypes
import glob
import os

import numpy as np
import torch

from sparse_executor_common import require_cuda
from sparse_npu_acl import scatter_csr_to_dense

CUSPARSE_STATUS_SUCCESS = 0

CUSPARSE_OPERATION_NON_TRANSPOSE = 0
CUSPARSE_OPERATION_TRANSPOSE = 1

CUSPARSE_INDEX_BASE_ZERO = 0

# cusparseIndexType_t: CUDA 11 uses INDEX_32I=0; CUDA 12+ uses INDEX_16U=1, INDEX_32I=2.
_INDEX_32I = None

# cusparseOrder_t: CUDA 11 uses COL=0/ROW=1; CUDA 12+ uses COL=1/ROW=2 (0 invalid).
_ORDER_ROW = None
_ORDER_COL = None


def _void_handle():
    return ctypes.c_void_p()


def _cuda_major_version():
    try:
        return int(str(torch.version.cuda).split('.')[0])
    except Exception:
        return 12


def _cusparse_index_32i():
    global _INDEX_32I
    if _INDEX_32I is not None:
        return _INDEX_32I
    _INDEX_32I = 2 if _cuda_major_version() >= 12 else 0
    return _INDEX_32I


def _cusparse_order_row_col():
    """Return (row-major, col-major) cusparseOrder_t values for this CUDA version."""
    global _ORDER_ROW, _ORDER_COL
    if _ORDER_ROW is not None:
        return _ORDER_ROW, _ORDER_COL
    if _cuda_major_version() >= 12:
        _ORDER_ROW, _ORDER_COL = 2, 1
    else:
        _ORDER_ROW, _ORDER_COL = 1, 0
    return _ORDER_ROW, _ORDER_COL


def _cusparse_dense_order(order_flag):
    row, col = _cusparse_order_row_col()
    return row if order_flag == 0 else col


def _cuda_dtype(name):
    return {'fp32': CUDA_R_32F, 'fp16': CUDA_R_16F, 'bf16': CUDA_R_16BF}[name]


def _torch_dtype(name):
    return {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[name]

CUSPARSE_FILL_MODE_LOWER = 0
CUSPARSE_FILL_MODE_UPPER = 1
CUSPARSE_DIAG_TYPE_NON_UNIT = 0
CUSPARSE_DIAG_TYPE_UNIT = 1

CUSPARSE_SPMAT_FILL_MODE = 0
CUSPARSE_SPMAT_DIAG_TYPE = 1

CUDA_R_32F = 0
CUDA_R_16F = 2
CUDA_R_16BF = 14

CUSPARSE_SPGEMM_DEFAULT = 0
CUSPARSE_SDDMM_ALG_DEFAULT = 0
CUSPARSE_SPSM_ALG_DEFAULT = 0
CUSPARSE_SPSM_UPDATE_GENERAL = 0

_LIB = None


def resolve_cusparse_lib():
    env = os.environ.get("CUSPARSE_LIB_PATH", "")
    if env and os.path.exists(env):
        return env
    try:
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        for name in ("libcusparse.so", "libcusparse.so.12", "libcusparse.so.11"):
            path = os.path.join(torch_lib, name)
            if os.path.exists(path):
                return path
    except Exception:
        pass
    for pattern in ("/usr/local/cuda/lib64/libcusparse.so*", "/usr/lib/x86_64-linux-gnu/libcusparse.so*"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return ""


def load_cusparse():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = resolve_cusparse_lib()
    if not path:
        raise RuntimeError("Cannot find libcusparse.so; set CUSPARSE_LIB_PATH")
    _LIB = ctypes.CDLL(path)
    _bind(_LIB)
    return _LIB


def check_cusparse(ret, op_name):
    if ret != CUSPARSE_STATUS_SUCCESS:
        raise RuntimeError(f"cusparse {op_name} failed: status={ret}")


def _bind(lib):
    c_void_p = ctypes.c_void_p
    c_int = ctypes.c_int
    c_int64 = ctypes.c_int64
    c_size_t = ctypes.c_size_t
    c_float = ctypes.c_float
    c_cuda_data_type = c_int

    lib.cusparseCreate.restype = c_int
    lib.cusparseCreate.argtypes = [ctypes.POINTER(c_void_p)]
    lib.cusparseDestroy.restype = c_int
    lib.cusparseDestroy.argtypes = [c_void_p]
    lib.cusparseSetStream.restype = c_int
    lib.cusparseSetStream.argtypes = [c_void_p, c_void_p]

    lib.cusparseCreateCsr.restype = c_int
    lib.cusparseCreateCsr.argtypes = [
        ctypes.POINTER(c_void_p), c_int64, c_int64, c_int64,
        c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int, c_cuda_data_type,
    ]
    lib.cusparseDestroySpMat.restype = c_int
    lib.cusparseDestroySpMat.argtypes = [c_void_p]
    lib.cusparseCsrSetPointers.restype = c_int
    lib.cusparseCsrSetPointers.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p]
    lib.cusparseSpMatGetSize.restype = c_int
    lib.cusparseSpMatGetSize.argtypes = [c_void_p, ctypes.POINTER(c_int64), ctypes.POINTER(c_int64), ctypes.POINTER(c_int64)]
    lib.cusparseSpMatSetAttribute.restype = c_int
    lib.cusparseSpMatSetAttribute.argtypes = [c_void_p, c_int, c_void_p, c_size_t]

    lib.cusparseCreateDnMat.restype = c_int
    lib.cusparseCreateDnMat.argtypes = [
        ctypes.POINTER(c_void_p), c_int64, c_int64, c_int64,
        c_void_p, c_cuda_data_type, c_int,
    ]
    lib.cusparseCreateConstDnMat.restype = c_int
    lib.cusparseCreateConstDnMat.argtypes = [
        ctypes.POINTER(c_void_p), c_int64, c_int64, c_int64,
        c_void_p, c_cuda_data_type, c_int,
    ]
    lib.cusparseDestroyDnMat.restype = c_int
    lib.cusparseDestroyDnMat.argtypes = [c_void_p]

    lib.cusparseSpGEMM_createDescr.restype = c_int
    lib.cusparseSpGEMM_createDescr.argtypes = [ctypes.POINTER(c_void_p)]
    lib.cusparseSpGEMM_destroyDescr.restype = c_int
    lib.cusparseSpGEMM_destroyDescr.argtypes = [c_void_p]
    lib.cusparseSpGEMM_workEstimation.restype = c_int
    lib.cusparseSpGEMM_workEstimation.argtypes = [
        c_void_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_cuda_data_type, c_int, c_void_p, ctypes.POINTER(c_size_t), c_void_p,
    ]
    lib.cusparseSpGEMM_compute.restype = c_int
    lib.cusparseSpGEMM_compute.argtypes = [
        c_void_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_cuda_data_type, c_int, c_void_p, ctypes.POINTER(c_size_t), c_void_p,
    ]
    lib.cusparseSpGEMM_copy.restype = c_int
    lib.cusparseSpGEMM_copy.argtypes = [
        c_void_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_cuda_data_type, c_int, c_void_p,
    ]

    lib.cusparseSDDMM_bufferSize.restype = c_int
    lib.cusparseSDDMM_bufferSize.argtypes = [
        c_void_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_cuda_data_type, c_int, ctypes.POINTER(c_size_t),
    ]
    lib.cusparseSDDMM_preprocess.restype = c_int
    lib.cusparseSDDMM_preprocess.argtypes = [
        c_void_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_cuda_data_type, c_int, c_void_p,
    ]
    lib.cusparseSDDMM.restype = c_int
    lib.cusparseSDDMM.argtypes = [
        c_void_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_cuda_data_type, c_int, c_void_p,
    ]

    lib.cusparseSpSM_createDescr.restype = c_int
    lib.cusparseSpSM_createDescr.argtypes = [ctypes.POINTER(c_void_p)]
    lib.cusparseSpSM_destroyDescr.restype = c_int
    lib.cusparseSpSM_destroyDescr.argtypes = [c_void_p]
    lib.cusparseSpSM_bufferSize.restype = c_int
    lib.cusparseSpSM_bufferSize.argtypes = [
        c_void_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p,
        c_cuda_data_type, c_int, c_void_p, ctypes.POINTER(c_size_t),
    ]
    lib.cusparseSpSM_analysis.restype = c_int
    lib.cusparseSpSM_analysis.argtypes = [
        c_void_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p,
        c_cuda_data_type, c_int, c_void_p, c_void_p,
    ]
    lib.cusparseSpSM_solve.restype = c_int
    lib.cusparseSpSM_solve.argtypes = [
        c_void_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p,
        c_cuda_data_type, c_int, c_void_p,
    ]
    lib.cusparseSpSM_updateMatrix.restype = c_int
    lib.cusparseSpSM_updateMatrix.argtypes = [c_void_p, c_void_p, c_void_p, c_int]


def _compute_cuda_dtype(ab_dtype_name, c_dtype_name):
    if ab_dtype_name == 'fp16' and c_dtype_name == 'fp32':
        return CUDA_R_32F
    if ab_dtype_name == 'fp16' and c_dtype_name == 'fp16':
        return CUDA_R_32F
    return _cuda_dtype(ab_dtype_name)


def _cusparse_order_row_col():
    """Detect valid cusparseOrder_t values once (CUDA version dependent)."""
    global _ORDER_ROW, _ORDER_COL
    if _ORDER_ROW is not None:
        return _ORDER_ROW, _ORDER_COL
    require_cuda()
    lib = load_cusparse()
    t = torch.zeros(2, 2, dtype=torch.float32, device='cuda')
    ptr = ctypes.c_void_p(t.data_ptr())
    # (row-major, col-major) pairs seen across CUDA releases
    candidates = ((2, 1), (1, 0), (0, 1))
    for row, col in candidates:
        d_row = ctypes.c_void_p()
        d_col = ctypes.c_void_p()
        ok_row = lib.cusparseCreateDnMat(
            ctypes.byref(d_row), 2, 2, 2, ptr, CUDA_R_32F, row) == CUSPARSE_STATUS_SUCCESS
        ok_col = lib.cusparseCreateDnMat(
            ctypes.byref(d_col), 2, 2, 2, ptr, CUDA_R_32F, col) == CUSPARSE_STATUS_SUCCESS
        if ok_row:
            lib.cusparseDestroyDnMat(d_row)
        if ok_col:
            lib.cusparseDestroyDnMat(d_col)
        if ok_row and ok_col:
            _ORDER_ROW, _ORDER_COL = row, col
            return _ORDER_ROW, _ORDER_COL
    raise RuntimeError('Cannot detect cusparseOrder_t row/col enum values for this libcusparse')


def _cusparse_dense_order(order_flag):
    row, col = _cusparse_order_row_col()
    return row if order_flag == 0 else col


def _cuda_dtype(name):
    return { 'fp32': CUDA_R_32F, 'fp16': CUDA_R_16F, 'bf16': CUDA_R_16BF }[name]


def _torch_dtype(name):
    return { 'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16 }[name]


def _safe(fn, *args):
    try:
        fn(*args)
    except Exception:
        pass


def _ptr(t):
    if t is None or t.numel() == 0:
        return ctypes.c_void_p(0)
    return ctypes.c_void_p(t.data_ptr())


def _h2d_int32(arr):
    if arr is None or arr.size == 0:
        return torch.empty(0, dtype=torch.int32, device='cuda')
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int32)).cuda()


def _h2d_vals(arr, dtype_name):
    dt = _torch_dtype(dtype_name)
    if arr is None or arr.size == 0:
        return torch.empty(0, dtype=dt, device='cuda')
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t.to(device='cuda', dtype=dt)


def _alloc_buf(nbytes):
    if nbytes <= 0:
        return None
    return torch.empty(nbytes, dtype=torch.uint8, device='cuda')


def _cuda_stream_ptr():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def cusparse_spgemm(
    m, k, n, alpha, beta,
    a_row, a_col, a_vals,
    b_row, b_col, b_vals,
    c_row=None, c_col=None, c_vals=None,
    dtype_name='fp32', alg=0, out_dtype=torch.float32,
    out_row_off=None, out_col_ind=None,
):
    """Run cusparseSpGEMM on GPU; return dense [m,n] for ATK comparison.

    cuSPARSE SpGEMM only accumulates beta*C when input and output C share the same
    sparsity pattern. When beta!=0 and patterns differ, compute alpha*A*B with
    beta=0 (using the golden output CSR as structure) then add beta*C on host.
    """
    require_cuda()
    add_beta_c = (
        beta != 0.0 and out_row_off is not None and out_col_ind is not None
        and c_vals is not None and c_vals.size > 0
    )
    beta_cusparse = 0.0 if add_beta_c else beta

    lib = load_cusparse()

    handle = _void_handle()
    matA = _void_handle()
    matB = _void_handle()
    matC = _void_handle()
    spgemm_descr = _void_handle()
    buf1 = buf2 = None

    dA_row = _h2d_int32(a_row)
    dA_col = _h2d_int32(a_col)
    dA_val = _h2d_vals(a_vals, dtype_name)
    dB_row = _h2d_int32(b_row)
    dB_col = _h2d_int32(b_col)
    dB_val = _h2d_vals(b_vals, dtype_name)

    nnz_a = int(a_vals.size)
    nnz_b = int(b_vals.size)

    if out_row_off is not None and out_col_ind is not None:
        c_row_host = np.ascontiguousarray(out_row_off, dtype=np.int32)
        c_col_host = np.ascontiguousarray(out_col_ind, dtype=np.int32)
        nnz_c = int(c_col_host.size)
        val_dt = np.float16 if dtype_name == 'fp16' else np.float32
        c_val_host = np.zeros(nnz_c, dtype=val_dt)
    else:
        c_row_host = np.zeros(m + 1, dtype=np.int32)
        c_col_host = np.zeros(0, dtype=np.int32)
        c_val_host = np.zeros(0, dtype=np.float32 if dtype_name != 'fp16' else np.float16)
        nnz_c = 0

    dC_row = _h2d_int32(c_row_host)
    dC_col = _h2d_int32(c_col_host)
    dC_val = _h2d_vals(c_val_host, dtype_name)

    opA = opB = CUSPARSE_OPERATION_NON_TRANSPOSE
    cuda_type = _cuda_dtype(dtype_name)
    cusparse_alg = CUSPARSE_SPGEMM_DEFAULT if alg == 0 else alg

    _alpha = ctypes.c_float(float(alpha))
    _beta = ctypes.c_float(float(beta_cusparse))
    alpha_p = ctypes.c_void_p(ctypes.addressof(_alpha))
    beta_p = ctypes.c_void_p(ctypes.addressof(_beta))

    try:
        check_cusparse(lib.cusparseCreate(ctypes.byref(handle)), "Create")
        check_cusparse(lib.cusparseSetStream(handle, _cuda_stream_ptr()), "SetStream")
        check_cusparse(lib.cusparseSpGEMM_createDescr(ctypes.byref(spgemm_descr)), "SpGEMM_createDescr")

        check_cusparse(lib.cusparseCreateCsr(
            ctypes.byref(matA), m, k, nnz_a, _ptr(dA_row), _ptr(dA_col), _ptr(dA_val),
            _cusparse_index_32i(), _cusparse_index_32i(), CUSPARSE_INDEX_BASE_ZERO, cuda_type), "CreateCsr(A)")
        check_cusparse(lib.cusparseCreateCsr(
            ctypes.byref(matB), k, n, nnz_b, _ptr(dB_row), _ptr(dB_col), _ptr(dB_val),
            _cusparse_index_32i(), _cusparse_index_32i(), CUSPARSE_INDEX_BASE_ZERO, cuda_type), "CreateCsr(B)")
        check_cusparse(lib.cusparseCreateCsr(
            ctypes.byref(matC), m, n, nnz_c, _ptr(dC_row), _ptr(dC_col), _ptr(dC_val),
            _cusparse_index_32i(), _cusparse_index_32i(), CUSPARSE_INDEX_BASE_ZERO, cuda_type), "CreateCsr(C)")

        buf_size1 = ctypes.c_size_t(0)
        check_cusparse(lib.cusparseSpGEMM_workEstimation(
            handle, opA, opB, alpha_p, matA, matB, beta_p, matC,
            cuda_type, cusparse_alg, spgemm_descr, ctypes.byref(buf_size1), None), "SpGEMM_workEstimation(size)")
        buf1 = _alloc_buf(buf_size1.value)
        check_cusparse(lib.cusparseSpGEMM_workEstimation(
            handle, opA, opB, alpha_p, matA, matB, beta_p, matC,
            cuda_type, cusparse_alg, spgemm_descr, ctypes.byref(buf_size1), _ptr(buf1)), "SpGEMM_workEstimation")

        buf_size2 = ctypes.c_size_t(0)
        check_cusparse(lib.cusparseSpGEMM_compute(
            handle, opA, opB, alpha_p, matA, matB, beta_p, matC,
            cuda_type, cusparse_alg, spgemm_descr, ctypes.byref(buf_size2), None), "SpGEMM_compute(size)")
        buf2 = _alloc_buf(buf_size2.value)
        check_cusparse(lib.cusparseSpGEMM_compute(
            handle, opA, opB, alpha_p, matA, matB, beta_p, matC,
            cuda_type, cusparse_alg, spgemm_descr, ctypes.byref(buf_size2), _ptr(buf2)), "SpGEMM_compute")

        rows = ctypes.c_int64()
        cols = ctypes.c_int64()
        nnz_out = ctypes.c_int64()
        check_cusparse(lib.cusparseSpMatGetSize(matC, ctypes.byref(rows), ctypes.byref(cols), ctypes.byref(nnz_out)), "SpMatGetSize")

        nnz_c_out = int(nnz_out.value)
        if nnz_c_out > dC_col.numel():
            dC_col = torch.empty(nnz_c_out, dtype=torch.int32, device='cuda')
            dC_val = torch.empty(nnz_c_out, dtype=_torch_dtype(dtype_name), device='cuda')
            if out_col_ind is not None and nnz_c_out <= len(out_col_ind):
                host_col = np.ascontiguousarray(out_col_ind[:nnz_c_out], dtype=np.int32)
                dC_col.copy_(torch.from_numpy(host_col).cuda())
            check_cusparse(lib.cusparseCsrSetPointers(
                matC, _ptr(dC_row), _ptr(dC_col), _ptr(dC_val)), "CsrSetPointers")
        elif nnz_c_out > 0:
            check_cusparse(lib.cusparseCsrSetPointers(
                matC, _ptr(dC_row), _ptr(dC_col), _ptr(dC_val)), "CsrSetPointers")

        check_cusparse(lib.cusparseSpGEMM_copy(
            handle, opA, opB, alpha_p, matA, matB, beta_p, matC,
            cuda_type, cusparse_alg, spgemm_descr), "SpGEMM_copy")

        torch.cuda.synchronize()

        h_row = dC_row.cpu().numpy()
        h_col = dC_col.cpu().numpy()[:nnz_c_out]
        h_val = dC_val.cpu().numpy()[:nnz_c_out]
        if dtype_name == 'fp16':
            h_val = h_val.astype(np.float32)
        elif dtype_name == 'bf16':
            h_val = torch.from_numpy(h_val.astype(np.float32)).to(torch.bfloat16).to(torch.float32).numpy()
        result = scatter_csr_to_dense(m, n, h_row, h_col, h_val, out_dtype)
        if add_beta_c:
            c_dense = scatter_csr_to_dense(m, n, c_row, c_col, c_vals, out_dtype)
            result = result + float(beta) * c_dense
        return result
    finally:
        _safe(lib.cusparseSpGEMM_destroyDescr, spgemm_descr)
        _safe(lib.cusparseDestroySpMat, matA)
        _safe(lib.cusparseDestroySpMat, matB)
        _safe(lib.cusparseDestroySpMat, matC)
        _safe(lib.cusparseDestroy, handle)
        del buf1, buf2


def _pad_dense_cuda(tensor, ld, rows, cols):
    if ld <= cols and tensor.is_contiguous() and tensor.shape[0] >= rows and tensor.shape[1] >= cols:
        return tensor[:rows, :cols].contiguous()
    out = torch.zeros(rows, ld, dtype=tensor.dtype, device=tensor.device)
    out[:, :cols] = tensor[:rows, :cols]
    return out


def cusparse_sddmm(
    m, n, k, alpha, beta,
    dense_a, dense_b, op_a, op_b, ld_a, ld_b, order_a, order_b,
    c_row, c_col, c_vals_host,
    ab_dtype_name, c_dtype_name, alg=0, out_dtype=torch.float32,
):
    """Run cusparseSDDMM on GPU; return dense [m,n] masked by CSR pattern."""
    require_cuda()
    lib = load_cusparse()
    handle = ctypes.c_void_p()
    mat_a = mat_b = mat_c = ctypes.c_void_p()
    buf = None

    ab_dt = _torch_dtype(ab_dtype_name)
    c_dt = _torch_dtype(c_dtype_name)
    if op_a == 0:
        a_rows, a_cols = m, k
    else:
        a_rows, a_cols = k, m
    ld_a_eff = ld_a if ld_a > 0 else (k if op_a == 0 else m)
    ld_b_eff = ld_b if ld_b > 0 else (n if op_b == 0 else k)
    b_rows, b_cols = (k, n) if op_b == 0 else (n, k)

    d_a = _pad_dense_cuda(dense_a.to(device='cuda', dtype=ab_dt), ld_a_eff, a_rows, a_cols)
    d_b = _pad_dense_cuda(dense_b.to(device='cuda', dtype=ab_dt), ld_b_eff, b_rows, b_cols)

    nnz = int(c_col.size)
    c_vals_np = c_vals_host.astype(np.float16 if c_dtype_name == 'fp16' else np.float32, copy=False)
    d_row = _h2d_int32(c_row)
    d_col = _h2d_int32(c_col)
    d_val = _h2d_vals(c_vals_np, c_dtype_name)

    op_a_c = CUSPARSE_OPERATION_NON_TRANSPOSE if op_a == 0 else CUSPARSE_OPERATION_TRANSPOSE
    op_b_c = CUSPARSE_OPERATION_NON_TRANSPOSE if op_b == 0 else CUSPARSE_OPERATION_TRANSPOSE
    order_a_c = _cusparse_dense_order(order_a)
    order_b_c = _cusparse_dense_order(order_b)
    ab_cuda_type = _cuda_dtype(ab_dtype_name)
    c_cuda_type = _cuda_dtype(c_dtype_name)
    compute_type = _compute_cuda_dtype(ab_dtype_name, c_dtype_name)

    _alpha = ctypes.c_float(float(alpha))
    _beta = ctypes.c_float(float(beta))
    alpha_p = ctypes.c_void_p(ctypes.addressof(_alpha))
    beta_p = ctypes.c_void_p(ctypes.addressof(_beta))
    cusparse_alg = CUSPARSE_SDDMM_ALG_DEFAULT if alg == 0 else alg

    try:
        check_cusparse(lib.cusparseCreate(ctypes.byref(handle)), "Create")
        check_cusparse(lib.cusparseSetStream(handle, _cuda_stream_ptr()), "SetStream")

        check_cusparse(lib.cusparseCreateDnMat(
            ctypes.byref(mat_a), a_rows, a_cols, ld_a_eff, _ptr(d_a), ab_cuda_type, order_a_c), "CreateDnMat(A)")
        check_cusparse(lib.cusparseCreateDnMat(
            ctypes.byref(mat_b), b_rows, b_cols, ld_b_eff, _ptr(d_b), ab_cuda_type, order_b_c), "CreateDnMat(B)")
        check_cusparse(lib.cusparseCreateCsr(
            ctypes.byref(mat_c), m, n, nnz, _ptr(d_row), _ptr(d_col), _ptr(d_val),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, c_cuda_type), "CreateCsr(C)")

        buf_size = ctypes.c_size_t(0)
        check_cusparse(lib.cusparseSDDMM_bufferSize(
            handle, op_a_c, op_b_c, alpha_p, mat_a, mat_b, beta_p, mat_c,
            compute_type, cusparse_alg, ctypes.byref(buf_size)), "SDDMM_bufferSize")
        buf = _alloc_buf(buf_size.value)
        check_cusparse(lib.cusparseSDDMM_preprocess(
            handle, op_a_c, op_b_c, alpha_p, mat_a, mat_b, beta_p, mat_c,
            compute_type, cusparse_alg, _ptr(buf)), "SDDMM_preprocess")
        check_cusparse(lib.cusparseSDDMM(
            handle, op_a_c, op_b_c, alpha_p, mat_a, mat_b, beta_p, mat_c,
            compute_type, cusparse_alg, _ptr(buf)), "SDDMM")

        torch.cuda.synchronize()
        h_val = d_val.cpu().numpy()
        if c_dtype_name == 'fp16':
            h_val = h_val.astype(np.float32)
        return scatter_csr_to_dense(m, n, c_row, c_col, h_val[:nnz], out_dtype)
    finally:
        _safe(lib.cusparseDestroyDnMat, mat_a)
        _safe(lib.cusparseDestroyDnMat, mat_b)
        _safe(lib.cusparseDestroySpMat, mat_c)
        _safe(lib.cusparseDestroy, handle)
        del buf


def _set_triangular_attrs(lib, mat_a, fill_mode, diag_type):
    fill = ctypes.c_int(CUSPARSE_FILL_MODE_LOWER if fill_mode == 0 else CUSPARSE_FILL_MODE_UPPER)
    diag = ctypes.c_int(CUSPARSE_DIAG_TYPE_UNIT if diag_type == 1 else CUSPARSE_DIAG_TYPE_NON_UNIT)
    check_cusparse(lib.cusparseSpMatSetAttribute(
        mat_a, CUSPARSE_SPMAT_FILL_MODE, ctypes.byref(fill), ctypes.sizeof(fill)), "SpMatSetAttribute(FILL)")
    check_cusparse(lib.cusparseSpMatSetAttribute(
        mat_a, CUSPARSE_SPMAT_DIAG_TYPE, ctypes.byref(diag), ctypes.sizeof(diag)), "SpMatSetAttribute(DIAG)")


def _create_spsm_dn_mats(lib, mat_b, mat_x, m, nrhs, b_ptr, c_ptr):
    """Create SpSM dense descriptors; matB is ConstDnMat, matC is DnMat (cuSPARSE §6.6.12)."""
    order_row, _ = _cusparse_order_row_col()
    check_cusparse(lib.cusparseCreateConstDnMat(
        ctypes.byref(mat_b), m, nrhs, nrhs, b_ptr, CUDA_R_32F, order_row), "CreateConstDnMat(B)")
    check_cusparse(lib.cusparseCreateDnMat(
        ctypes.byref(mat_x), m, nrhs, nrhs, c_ptr, CUDA_R_32F, order_row), "CreateDnMat(C)")


def cusparse_spsm(
    m, nrhs, alpha,
    a_row, a_col, a_vals,
    dense_b, dense_c_out,
    op_a, op_b, fill_mode, diag_type,
    in_place=False, null_values=False, update_matrix=False,
    alg=0, out_dtype=torch.float32,
):
    """Run cusparseSpSM on GPU; return dense [m, nrhs].

    Flow aligned with NPU aclsparseSpSM* and cuSPARSE §6.6.12:
      createDescr -> bufferSize -> analysis -> [recreate B/C if null_values] -> solve
      -> [updateMatrix + second solve if update_matrix]
    """
    require_cuda()
    lib = load_cusparse()

    handle = ctypes.c_void_p()
    spsm_descr = ctypes.c_void_p()
    mat_a = mat_b = mat_x = ctypes.c_void_p()
    buf = None
    d_new_vals = None

    d_row = _h2d_int32(a_row)
    d_col = _h2d_int32(a_col)
    d_val = _h2d_vals(a_vals, 'fp32')
    nnz = int(a_vals.size)

    # Caller applies op(B) on host layout (same as NPU executor _apply_op_b).
    b_in = dense_b.contiguous().cuda()
    x_out = b_in if in_place else dense_c_out.contiguous().cuda()

    op_a_c = CUSPARSE_OPERATION_NON_TRANSPOSE if op_a == 0 else CUSPARSE_OPERATION_TRANSPOSE
    op_b_c = CUSPARSE_OPERATION_NON_TRANSPOSE if op_b == 0 else CUSPARSE_OPERATION_TRANSPOSE
    cusparse_alg = CUSPARSE_SPSM_ALG_DEFAULT if alg == 0 else alg

    _alpha = ctypes.c_float(float(alpha))
    alpha_p = ctypes.c_void_p(ctypes.addressof(_alpha))

    try:
        check_cusparse(lib.cusparseCreate(ctypes.byref(handle)), "Create")
        check_cusparse(lib.cusparseSetStream(handle, _cuda_stream_ptr()), "SetStream")
        check_cusparse(lib.cusparseSpSM_createDescr(ctypes.byref(spsm_descr)), "SpSM_createDescr")

        check_cusparse(lib.cusparseCreateCsr(
            ctypes.byref(mat_a), m, m, nnz, _ptr(d_row), _ptr(d_col), _ptr(d_val),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), "CreateCsr(A)")
        _set_triangular_attrs(lib, mat_a, fill_mode, diag_type)

        # TC-08 / nullValues: bufferSize & analysis allow NULL matB/matC values pointers;
        # descriptors must still be valid (rows/cols/ld/type). Real pointers required at solve.
        b_ptr = ctypes.c_void_p(0 if null_values else b_in.data_ptr())
        c_ptr = ctypes.c_void_p(0 if null_values else x_out.data_ptr())
        _create_spsm_dn_mats(lib, mat_b, mat_x, m, nrhs, b_ptr, c_ptr)

        buf_size = ctypes.c_size_t(0)
        check_cusparse(lib.cusparseSpSM_bufferSize(
            handle, op_a_c, op_b_c, alpha_p, mat_a, mat_b, mat_x,
            CUDA_R_32F, cusparse_alg, spsm_descr, ctypes.byref(buf_size)), "SpSM_bufferSize")
        buf = _alloc_buf(buf_size.value)
        check_cusparse(lib.cusparseSpSM_analysis(
            handle, op_a_c, op_b_c, alpha_p, mat_a, mat_b, mat_x,
            CUDA_R_32F, cusparse_alg, spsm_descr, _ptr(buf)), "SpSM_analysis")

        if null_values:
            _safe(lib.cusparseDestroyDnMat, mat_b)
            _safe(lib.cusparseDestroyDnMat, mat_x)
            mat_b = mat_x = ctypes.c_void_p()
            b_ptr = ctypes.c_void_p(b_in.data_ptr())
            c_ptr = ctypes.c_void_p(x_out.data_ptr())
            _create_spsm_dn_mats(lib, mat_b, mat_x, m, nrhs, b_ptr, c_ptr)

        check_cusparse(lib.cusparseSpSM_solve(
            handle, op_a_c, op_b_c, alpha_p, mat_a, mat_b, mat_x,
            CUDA_R_32F, cusparse_alg, spsm_descr), "SpSM_solve")

        # TC-04 / updateMatrix: refresh sparse A values in spsmDescr, then re-solve.
        if update_matrix:
            new_vals = a_vals.copy()
            new_vals = new_vals + np.random.RandomState(0).uniform(
                -0.1, 0.1, size=new_vals.shape).astype(np.float32)
            d_new_vals = _h2d_vals(new_vals, 'fp32')
            check_cusparse(lib.cusparseSpSM_updateMatrix(
                handle, spsm_descr, _ptr(d_new_vals), CUSPARSE_SPSM_UPDATE_GENERAL),
                "SpSM_updateMatrix")
            check_cusparse(lib.cusparseSpSM_solve(
                handle, op_a_c, op_b_c, alpha_p, mat_a, mat_b, mat_x,
                CUDA_R_32F, cusparse_alg, spsm_descr), "SpSM_solve(after updateMatrix)")

        torch.cuda.synchronize()
        return x_out.cpu().to(out_dtype)
    finally:
        _safe(lib.cusparseSpSM_destroyDescr, spsm_descr)
        _safe(lib.cusparseDestroySpMat, mat_a)
        _safe(lib.cusparseDestroyDnMat, mat_b)
        _safe(lib.cusparseDestroyDnMat, mat_x)
        _safe(lib.cusparseDestroy, handle)
        del buf, d_new_vals
